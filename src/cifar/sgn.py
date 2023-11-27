# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Code to train the Shifted Gaussian Noise (SGN) method with a Wide ResNet 28-2 on CIFAR-10/100.
    Most relevant parts of the code are:
    - Implementations of log-ratio transforms, starting on line 119.
    - The function to create a shifted Gaussian distribution on line 149.
    - The training step function on line 448.

    The code uses and is based on the Uncertainty Baselines GitHub repo:
    https://github.com/google/uncertainty-baselines
"""


import os
import sys
import time
from absl import app
from absl import flags
from absl import logging

import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import uncertainty_baselines as ub
import utils  # local file import
import ood_utils  # local file import
from label_corrupted_dataset import make_label_corrupted_dataset # local file import
from wide_resnet_factors import wide_resnet # local file import


flags.register_validator('train_proportion',
                         lambda tp: tp > 0.0 and tp <= 1.0,
                         message='--train_proportion must be in (0, 1].')
flags.DEFINE_float('label_smoothing', 0.001, 'Label smoothing parameter in (0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls > 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in (0, 1].')

# Fine-grained specification of the hyperparameters (used when FLAGS.l2 is None)
flags.DEFINE_float('bn_l2', None, 'L2 reg. coefficient for batch-norm layers.')
flags.DEFINE_float('input_conv_l2', None,
                   'L2 reg. coefficient for the input conv layer.')
flags.DEFINE_float('group_1_conv_l2', None,
                   'L2 reg. coefficient for the 1st group of conv layers.')
flags.DEFINE_float('group_2_conv_l2', None,
                   'L2 reg. coefficient for the 2nd group of conv layers.')
flags.DEFINE_float('group_3_conv_l2', None,
                   'L2 reg. coefficient for the 3rd group of conv layers.')
flags.DEFINE_float('dense_kernel_l2', None,
                   'L2 reg. coefficient for the kernel of the dense layer.')
flags.DEFINE_float('dense_bias_l2', None,
                   'L2 reg. coefficient for the bias of the dense layer.')
flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')


# SGN flags
flags.DEFINE_float(
    'alpha', 0.995, 'Exponential moving average weight for label estimation.')
flags.DEFINE_float(
    'beta', 0.9999, 'Exponential moving average weight for parameters of delta model.')


# OOD flags.
flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_integer('ood_interval', 0,
                  'Interval to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
                  'list of OOD datasets to evaluate on.')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
                  'Wheter to use DempsterShafer Uncertainty score.')


# Noisy labels flags.
flags.DEFINE_bool('noisy_labels', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_float('severity', 0.0,
                   'Noise ratio..')
flags.DEFINE_enum('corruption_type', 'asym',
                  enum_values=['sym', 'asym', 'worst', 'aggre', 'rand1', 'rand2', 'rand3', 'c100noise', 'instance'],
                  help='Type of label noise.')



FLAGS = flags.FLAGS

def get_dim_logits(num_classes):
    return num_classes-1


def get_smoothed_onehot(labels, num_classes):
    """ Implements standard label smoothing. """
    dim = num_classes
    one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), dim)
    smoothed_targets = (1.0-FLAGS.label_smoothing) * one_hot_labels + \
        FLAGS.label_smoothing * tf.ones(tf.shape(one_hot_labels)) / float(dim)
    return smoothed_targets

# ---------------------------------------Log-Ratio Transforms--------------------------------------
def clr_inv(p):
    z = tf.math.log(p)
    return z - tf.reduce_mean(z, axis=1)[:, tf.newaxis]


def clr_forward(z, axis=1):
    return tf.nn.softmax(z, axis=axis)


def helmert_tf(n):
  tensor = tf.ones((n, n))
  H = tf.linalg.set_diag(tf.linalg.band_part(tensor, -1, 0), 1-tf.range(1, n+1, dtype=tf.float32))
  d = tf.range(0, n, dtype=tf.float32) * tf.range(1, n+1, dtype=tf.float32)
  H_full = H / tf.math.sqrt(d)[:, tf.newaxis]
  return H_full[1:]


def ilr_forward(z, axis=-1):
    H = helmert_tf(tf.shape(z)[-1] + 1)
    return clr_forward(z @ H, axis=axis)


def ilr_inv(p):
    z = clr_inv(p)
    H = helmert_tf(tf.shape(p)[-1])
    return z @ tf.linalg.matrix_transpose(H)
# -------------------------------------------------------------------------------------------------


def _create_normal(mu, r, mu_ema, num_classes, labels, exponent):
    """
    Utility function for creating a shifted Gaussian distribution in logit space.

    Arguments:
      mu: The unshifted mean predicted by the main network.
      r: The rank-1 factor of the scale matrix.
      mu_ema: The predicted mean of the EMA network used to calculate the shift: delta = t - mu_ema.
      num_classes: The number of classes for the dataset, i.e. 10 or 100.
      labels: Soft labels from the use of label smoothing.
      exponent: The current step of the optimization, used as exponent scale factor of delta.
    
    Returns:
      tfp.distributions.MultivariateNormalDiagPlusLowRank.
    """

    num_classes_logits = get_dim_logits(num_classes)

    mean = mu
    diag = tf.ones([tf.shape(mu)[0], num_classes_logits])
    r = tf.reshape(r, [-1, num_classes_logits, 1])

    mu_ema = tf.stop_gradient(mu_ema)
    factor = 1.0-tf.math.pow(FLAGS.alpha, exponent)

    t = ilr_inv(labels)
    mean += factor * (t - mu_ema)

    return tfd.MultivariateNormalDiagPlusLowRank(loc=mean,
                                                  scale_diag=diag,
                                                  scale_perturb_factor=r,
                                                  validate_args=False,
                                                  allow_nan_stats=False)



def _extract_hyperparameter_dictionary():
  """Create the dictionary of hyperparameters from FLAGS."""
  flags_as_dict = FLAGS.flag_values_dict()
  hp_keys = ub.models.models.wide_resnet.HP_KEYS
  hps = {k: flags_as_dict[k] for k in hp_keys}
  return hps


def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  data_dir = FLAGS.data_dir

  # Use GPU
  strategy = tf.distribute.MirroredStrategy()

  ds_info = tfds.builder(FLAGS.dataset, data_dir=data_dir).info
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  train_dataset_size = (
      ds_info.splits['train'].num_examples * FLAGS.train_proportion)
  steps_per_epoch = int(train_dataset_size / batch_size)
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', ds_info.splits['train'].num_examples)
  logging.info('Train proportion %s', FLAGS.train_proportion)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      validation_percent=1. - FLAGS.train_proportion)


  if FLAGS.noisy_labels:
    # Create noisy dataset by augmenting clean with extra noisy label
    dataset_cls = ub.datasets.DATASETS[FLAGS.dataset]
    dataset_cls = make_label_corrupted_dataset(dataset_cls)
    train_builder = dataset_cls(
          dataset=train_builder,
          split=tfds.Split.TRAIN,
          data_dir=data_dir,
          seed=FLAGS.seed,
          shuffle_buffer_size=FLAGS.shuffle_buffer_size,
          validation_percent=1.0 - FLAGS.train_proportion,
          corruption_type=FLAGS.corruption_type,
          severity=FLAGS.severity)


  train_dataset = train_builder.load(batch_size=batch_size)
  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:
    validation_builder = ub.datasets.get(
        FLAGS.dataset,
        data_dir=data_dir,
        split=tfds.Split.VALIDATION,
        validation_percent=1. - FLAGS.train_proportion)
    if FLAGS.noisy_labels:
      dataset_cls = ub.datasets.DATASETS[FLAGS.dataset]
      dataset_cls = make_label_corrupted_dataset(dataset_cls)
      validation_builder = dataset_cls(
              dataset=validation_builder,
              split=tfds.Split.VALIDATION,
              data_dir=data_dir,
              seed=FLAGS.seed,
              validation_percent=1.0 - FLAGS.train_proportion,
              corruption_type=FLAGS.corruption_type,
              severity=FLAGS.severity)

    validation_dataset = validation_builder.load(batch_size=batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // batch_size
  clean_test_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      split=tfds.Split.TEST)
  clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_builder.num_examples // batch_size
  num_classes = 100 if FLAGS.dataset == 'cifar100' else 10
  dim_logits = get_dim_logits(num_classes)

  if FLAGS.ood_interval > 0:
    ood_dataset_names = FLAGS.ood_dataset
    ood_ds, steps_per_ood = ood_utils.load_ood_datasets(
        ood_dataset_names, clean_test_builder, 1. - FLAGS.train_proportion,
        batch_size,
        download_data=FLAGS.download_data)
    ood_datasets = {
        name: strategy.experimental_distribute_dataset(ds)
        for name, ds in ood_ds.items()
    }

  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar100':
      data_dir = FLAGS.cifar100_c_path
    corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
    for corruption_type in corruption_types:
      for severity in range(1, 6):
        dataset = ub.datasets.get(
            f'{FLAGS.dataset}_corrupted',
            corruption_type=corruption_type,
            data_dir=data_dir,
            severity=severity,
            split=tfds.Split.TEST).load(batch_size=batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))


  with strategy.scope():
    logging.info('Building ResNet model')
    model = wide_resnet(
      input_shape=(32, 32, 3),
      depth=28,
      width_multiplier=FLAGS.width,
      num_classes=dim_logits,
      l2=FLAGS.l2,
      hps=_extract_hyperparameter_dictionary(),
      version=2,
      num_factors=1,
      no_scale=False)

    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.beta)
    model_delta = wide_resnet(
      input_shape=(32, 32, 3),
      depth=28,
      width_multiplier=FLAGS.width,
      num_classes=dim_logits,
      l2=FLAGS.l2,
      hps=_extract_hyperparameter_dictionary(),
      version=2,
      num_factors=1,
      no_scale=False)

    ema.apply(model.trainable_variables)



    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    optimizer = tf.keras.optimizers.SGD(base_lr,
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)

    metrics = {
        'train/accuracy_nl':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/negative_log_likelihood_c10n':
            tf.keras.metrics.Mean(),
        'train/accuracy_tl_clean':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_tl_corrupted':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_nl_corrupted':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_tl':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_delta_tl_clean':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_delta_tl_corrupted':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_delta_nl_corrupted':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_delta':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_delta_tl':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
        'train/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood_delta':
            tf.keras.metrics.Mean(),
        'test/accuracy_delta':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece_delta':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),

    }
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy_tl': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/accuracy_nl': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
          'validation/negative_log_likelihood_delta': tf.keras.metrics.Mean(),
          'validation/accuracy_tl_delta': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/accuracy_nl_delta': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece_delta': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
      })

    if FLAGS.ood_interval > 0:
      ood_metrics = ood_utils.create_ood_metrics(ood_dataset_names)
      metrics.update(ood_metrics)

    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, 6):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_delta = tf.train.Checkpoint(model=model_delta, optimizer=optimizer)

    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    total_steps = steps_per_epoch * FLAGS.train_epochs

    if FLAGS.saved_model_dir:
      logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
    if FLAGS.eval_only:
      initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval


  @tf.function
  def train_step(iterator, epoch):
    """Training StepFn."""
    def step_fn(inputs, step):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['noisy_labels'] if FLAGS.noisy_labels else inputs['labels']
      smoothed_targets = get_smoothed_onehot(labels, num_classes)
      logit_targets = ilr_inv(smoothed_targets)

      loc_ema, _ = model_delta(images, training=True)

      exponent = tf.cast(epoch, tf.float32) + \
                tf.cast(step, tf.float32) / steps_per_epoch


      with tf.GradientTape(persistent=True) as tape:
        loc, scale = model(images, training=True)
      
        normal = _create_normal(loc, scale, loc_ema, num_classes,
                                smoothed_targets, exponent)

        ll_vec = normal.log_prob(logit_targets)
        negative_log_likelihood = -tf.reduce_mean(ll_vec)

        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      opt_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

      with tf.control_dependencies([opt_op]):
        ema.apply(model.trainable_variables)
        # Update the weights of the ema model
        for v, v_delta in zip(model.trainable_variables, model_delta.trainable_variables):
          v_delta.assign(ema.average(v))

      probs = ilr_forward(loc, axis=1)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/accuracy_tl'].update_state(inputs['labels'], probs)

      probs_delta = ilr_forward(loc_ema, axis=1)
      metrics['train/accuracy_delta'].update_state(labels, probs_delta)
      metrics['train/accuracy_delta_tl'].update_state(inputs['labels'], probs_delta)


      if FLAGS.noisy_labels:
          mask = inputs['labels'] == inputs['noisy_labels']
          tl_labels_clean, tl_labels_corrupted = tf.boolean_mask(inputs['labels'], mask), tf.boolean_mask(inputs['labels'], ~mask)
          nl_labels_corrupted = tf.boolean_mask(inputs['noisy_labels'], ~mask)
          probs_clean, probs_corrupted = tf.boolean_mask(probs, mask), tf.boolean_mask(probs, ~mask)

          # Accuracy
          metrics['train/accuracy_tl_clean'].update_state(tl_labels_clean, probs_clean)
          metrics['train/accuracy_tl_corrupted'].update_state(tl_labels_corrupted, probs_corrupted)
          metrics['train/accuracy_nl_corrupted'].update_state(nl_labels_corrupted, probs_corrupted)

          probs_delta_clean, probs_delta_corrupted = tf.boolean_mask(probs_delta, mask), tf.boolean_mask(probs_delta, ~mask)

          # Accuracy
          metrics['train/accuracy_delta_tl_clean'].update_state(tl_labels_clean, probs_delta_clean)
          metrics['train/accuracy_delta_tl_corrupted'].update_state(tl_labels_corrupted, probs_delta_corrupted)
          metrics['train/accuracy_delta_nl_corrupted'].update_state(nl_labels_corrupted, probs_delta_corrupted)


    for step in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator), step,))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps, model_eval, suffix):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      labels_noisy = inputs['noisy_labels'] if FLAGS.noisy_labels and dataset_split in ['train', 'validation'] else inputs['labels']

      loc, _ = model_eval(images, training=False)
 
      probs = ilr_forward(loc, axis=1)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        if dataset_split == 'validation':
          metrics[f'{dataset_split}/negative_log_likelihood' + suffix].update_state(
            negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy_tl' + suffix].update_state(labels, probs)
          metrics[f'{dataset_split}/accuracy_nl' + suffix].update_state(labels_noisy, probs)
          metrics[f'{dataset_split}/ece' + suffix].add_batch(probs, label=labels)
        else:
          metrics[f'{dataset_split}/negative_log_likelihood' + suffix].update_state(
              negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy' + suffix].update_state(labels, probs)
          metrics[f'{dataset_split}/ece' + suffix].add_batch(probs, label=labels)
      elif dataset_name == 'val':
        assert False
        metrics[f'validation/negative_log_likelihood'].update_state(
          negative_log_likelihood)
        metrics[f'validation/accuracy_tl'].update_state(labels, probs)
        metrics[f'validation/accuracy_nl'].update_state(labels_noisy, probs)
        metrics[f'validation/ece'].add_batch(probs, label=labels)
      elif dataset_name == 'train_c10n' and FLAGS.noisy_labels and FLAGS.eval_only:
        assert False
        rand1, rand2, rand3 = inputs['label_rand1'], inputs['label_rand2'], inputs['label_rand3']
        nll_vec = (tf.keras.losses.sparse_categorical_crossentropy(rand1, probs) +
                   tf.keras.losses.sparse_categorical_crossentropy(rand2, probs) +
                   tf.keras.losses.sparse_categorical_crossentropy(rand3, probs))
        negative_log_likelihood = tf.reduce_mean(nll_vec) / 3.0
        metrics[f'{dataset_split}/negative_log_likelihood_c10n'].update_state(
          negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy_nl'].update_state(labels_noisy, probs)
      elif dataset_name.startswith('ood'):
        assert False
        ood_labels = 1 - inputs['is_in_distribution']
        if FLAGS.dempster_shafer_ood:
          probs_clipped = tf.clip_by_value(probs, 1e-7, 1.0)
          log_probs = tf.math.log(probs_clipped)
          ood_scores = ood_utils.DempsterShaferUncertainty(log_probs)
        else:
          ood_scores = 1 - tf.reduce_max(probs, axis=-1)

        # Edgecase for if dataset_name contains underscores
        ood_dataset_name = '_'.join(dataset_name.split('_')[1:])
        for name, metric in metrics.items():
          if ood_dataset_name in name:
            metric.update_state(ood_labels, ood_scores)
      elif FLAGS.corruptions_interval > 0:
        assert False
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})

  train_iterator = iter(train_dataset)
  start_time = time.time()
  tb_callback = None
  if FLAGS.collect_profile:
    tb_callback = tf.keras.callbacks.TensorBoard(
        profile_batch=(100, 102),
        log_dir=os.path.join(FLAGS.output_dir, 'logs'))
    tb_callback.set_model(model)

  models = [model, model_delta]
  suffixes = ['', '_delta']

  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)

    if not FLAGS.eval_only:
      train_start_time = time.time()
      epoch_tensor = tf.convert_to_tensor(epoch, tf.int64)
      train_step(train_iterator, epoch_tensor)
      ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
      metrics['train/ms_per_example'].update_state(ms_per_example)

      current_step = (epoch + 1) * steps_per_epoch
      max_steps = steps_per_epoch * FLAGS.train_epochs
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                     current_step / max_steps,
                     epoch + 1,
                     FLAGS.train_epochs,
                     steps_per_sec,
                     eta_seconds / 60,
                     time_elapsed / 60))
      logging.info(message)
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    if FLAGS.eval_only:
      test_step(train_iterator, 'train', 'train_c10n', steps_per_epoch)



    for model_eval, suffix in zip(models, suffixes):
      if validation_dataset:
        validation_iterator = iter(validation_dataset)
        test_step(
            validation_iterator, 'validation', 'clean', steps_per_validation, model_eval, suffix)
      datasets_to_evaluate = {'clean': test_datasets['clean']}
      if (FLAGS.corruptions_interval > 0 and
          (epoch + 1) % FLAGS.corruptions_interval == 0):
          datasets_to_evaluate = test_datasets
      for dataset_name, test_dataset in datasets_to_evaluate.items():
        test_iterator = iter(test_dataset)
        logging.info('Testing on dataset %s', dataset_name)
        logging.info('Starting to run eval at epoch: %s', epoch)
        test_start_time = time.time()
        test_step(test_iterator, 'test', dataset_name, steps_per_eval, model_eval, suffix)
        ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
        metrics['test/ms_per_example'].update_state(ms_per_example)

        logging.info('Done with testing on %s', dataset_name)

      if (FLAGS.ood_interval > 0 and
          (epoch + 1) % FLAGS.ood_interval == 0):

        for dataset_name in ood_dataset_names:
          ood_iterator = iter(ood_datasets['ood_{}'.format(dataset_name)])
          logging.info('Calculating OOD on dataset %s', dataset_name)
          logging.info('Running OOD eval at epoch: %s', epoch)
          test_step(ood_iterator, 'test', 'ood_{}'.format(dataset_name),
                    steps_per_ood[dataset_name], model_eval, suffix)

          logging.info('Done with OOD eval on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(corrupt_results)
    # Metrics from Robustness Metrics (like ECE) will return a dict with a
    # single key/value, instead of a scalar.
    total_results = {
        k: (list(v.values())[0] if isinstance(v, dict) else v)
        for k, v in total_results.items()
    }
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)

if __name__ == '__main__':
  app.run(main)
