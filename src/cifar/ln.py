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

""" Logit-Normal Wide ResNet 28-2 on CIFAR-10/100 trained with MLE.
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
flags.DEFINE_float('label_smoothing', 0.1, 'Label smoothing parameter in (0,1].')
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

# Heteroscedastic flags.
flags.DEFINE_float('temperature', 1.0, # 1.3
                   'Temperature for heteroscedastic head.')
flags.DEFINE_float('min_scale', 1.0,
                   'Added constant to the predicted scale parameters.')
flags.DEFINE_float('max_scale', 1.0,
                   'Added constant to the predicted scale parameters.')
flags.DEFINE_bool('mu_as_pred', False,
                  'Whether to use mu instead of average of samples as pred.')
flags.DEFINE_float('grad_clip_norm', -1.0, 'Global gradient clipping')


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


def _create_logitnormal(loc, scale, min_scale, num_classes, labels=None):
  # Create the Logit-Normal Distribution
  loc = tf.ensure_shape(loc, [None, num_classes])
  scale = tf.ensure_shape(scale, [None, num_classes])
  scale = tf.reshape(scale, [-1, num_classes, 1])
  diag = min_scale * tf.ones([tf.shape(loc)[0], num_classes])

  mvn = tfd.MultivariateNormalDiagPlusLowRank(loc=loc,
                                              scale_diag=diag,
                                              scale_perturb_factor=scale,
                                              validate_args=False,
                                              allow_nan_stats=False) # Debug

  tf.debugging.assert_all_finite(scale, "Scale contains NaN values")
  bijector = tfb.Chain([tfb.SoftmaxCentered(), tfb.Scale(1.0 / FLAGS.temperature)])
  

  logit_normal = tfd.TransformedDistribution(
    distribution=mvn,
    bijector=bijector,
    name='LogitNormalTransformedDistribution')

  return logit_normal


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
      num_classes=num_classes,
      l2=FLAGS.l2,
      hps=_extract_hyperparameter_dictionary(),
      version=2,
      num_factors=1,
      no_scale=False)


    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

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
        'train/entropy_clean':
            tf.keras.metrics.Mean(),
        'train/entropy_noisy':
            tf.keras.metrics.Mean(),
        'train/mse_mu_label_clean':
            tf.keras.metrics.Mean(),
        'train/mse_mu_label_noisy':
            tf.keras.metrics.Mean(),
        'train/grad_scale':
            tf.keras.metrics.Mean(),
        'train/grad_mu':
            tf.keras.metrics.Mean(),
        'train/grad_scale_clean':
            tf.keras.metrics.Mean(),
        'train/grad_scale_noisy':
            tf.keras.metrics.Mean(),
        'train/grad_mu_clean':
            tf.keras.metrics.Mean(),
        'train/grad_mu_noisy':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/accuracy_tl':
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
    }
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy_tl': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/accuracy_nl': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
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
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    #min_scale = FLAGS.max_scale
    total_steps = steps_per_epoch * FLAGS.train_epochs

    if FLAGS.saved_model_dir:
      logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
    if FLAGS.eval_only:
      initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval


  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['noisy_labels'] if FLAGS.noisy_labels else inputs['labels']
    
      with tf.GradientTape(persistent=True) as tape:

        loc, scale = model(images, training=True)
      
        t = tf.cast(optimizer.iterations / total_steps, tf.float32)
        min_scale = FLAGS.max_scale * (1.0 - t) + FLAGS.min_scale * t

        one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes+1)
        smoothed_targets = (1.0-FLAGS.label_smoothing) * one_hot_labels + \
          FLAGS.label_smoothing * tf.ones(tf.shape(one_hot_labels)) / (num_classes+1)


        logit_normal = _create_logitnormal(loc, scale, min_scale, num_classes, smoothed_targets)
        nll_vec = logit_normal.log_prob(smoothed_targets)
        negative_log_likelihood = -tf.reduce_mean(nll_vec)

        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      if not FLAGS.grad_clip_norm < 0.0:
          grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.grad_clip_norm)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))


      grads_mu = tape.gradient(scaled_loss, loc)
      grads_scale = tape.gradient(scaled_loss, scale)
      grads_mu = tf.ensure_shape(grads_mu, [None, num_classes])
      grads_scale = tf.ensure_shape(grads_scale, [None, num_classes])

      grads_mu = tf.norm(grads_mu, axis=1)
      grads_scale = tf.norm(grads_scale, axis=1)
      grads_mu = tf.ensure_shape(grads_mu, [None])
      grads_scale = tf.ensure_shape(grads_scale, [None])


      probs = logit_normal.bijector.forward(loc, axis=1)

      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/accuracy_tl'].update_state(inputs['labels'], probs)

      metrics['train/grad_mu'].update_state(tf.reduce_sum(grads_mu))
      metrics['train/grad_scale'].update_state(tf.reduce_sum(grads_scale))


      if FLAGS.noisy_labels:
          mask = inputs['labels'] == inputs['noisy_labels']
          tl_labels_clean, tl_labels_corrupted = tf.boolean_mask(inputs['labels'], mask), tf.boolean_mask(inputs['labels'], ~mask)
          nl_labels_corrupted = tf.boolean_mask(inputs['noisy_labels'], ~mask)
          probs_clean, probs_corrupted = tf.boolean_mask(probs, mask), tf.boolean_mask(probs, ~mask)

          # Accuracy
          metrics['train/accuracy_tl_clean'].update_state(tl_labels_clean, probs_clean)
          metrics['train/accuracy_tl_corrupted'].update_state(tl_labels_corrupted, probs_corrupted)
          metrics['train/accuracy_nl_corrupted'].update_state(nl_labels_corrupted, probs_corrupted)

          entropy = logit_normal.distribution.entropy()
          entropy_c, entropy_n = tf.reduce_mean(tf.boolean_mask(entropy, mask)), tf.reduce_mean(tf.boolean_mask(entropy, ~mask))
        
          target_logits = logit_normal.bijector.inverse(smoothed_targets, axis=1)
          mse_mu = tf.reduce_mean(tf.math.square(loc - target_logits), axis=1)

          mse_mu_c, mse_mu_n = tf.reduce_mean(tf.boolean_mask(mse_mu, mask)), tf.reduce_mean(tf.boolean_mask(mse_mu, ~mask))
          g_mu_c, g_mu_n = tf.reduce_sum(tf.boolean_mask(grads_mu, mask)), tf.reduce_sum(tf.boolean_mask(grads_mu, ~mask))
          g_scale_c, g_scale_n = tf.reduce_sum(tf.boolean_mask(grads_scale, mask)), tf.reduce_sum(tf.boolean_mask(grads_scale, ~mask))

          metrics['train/entropy_clean'].update_state(entropy_c)
          metrics['train/entropy_noisy'].update_state(entropy_n)
          metrics['train/mse_mu_label_clean'].update_state(mse_mu_c)
          metrics['train/mse_mu_label_noisy'].update_state(mse_mu_n)
          metrics['train/grad_mu_clean'].update_state(tf.reduce_mean(g_mu_c))
          metrics['train/grad_mu_noisy'].update_state(tf.reduce_mean(g_mu_n))
          metrics['train/grad_scale_clean'].update_state(tf.reduce_mean(g_scale_c))
          metrics['train/grad_scale_noisy'].update_state(tf.reduce_mean(g_scale_n))


    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      labels_noisy = inputs['noisy_labels'] if FLAGS.noisy_labels and dataset_split in ['train', 'validation'] else inputs['labels']

      loc, scale = model(images, training=False)
 

      t = tf.cast(optimizer.iterations / total_steps, tf.float32)
      min_scale = FLAGS.max_scale * (1.0 - t) + FLAGS.min_scale * t
      logit_normal = _create_logitnormal(loc, scale, min_scale, num_classes)


      probs = logit_normal.bijector.forward(loc, axis=1)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        if dataset_split == 'validation':
          metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
            negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy_tl'].update_state(labels, probs)
          metrics[f'{dataset_split}/accuracy_nl'].update_state(labels_noisy, probs)
          metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
        else:
          metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
              negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
          metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
      elif dataset_name == 'val':
        metrics[f'validation/negative_log_likelihood'].update_state(
          negative_log_likelihood)
        metrics[f'validation/accuracy_tl'].update_state(labels, probs)
        metrics[f'validation/accuracy_nl'].update_state(labels_noisy, probs)
        metrics[f'validation/ece'].add_batch(probs, label=labels)
      elif dataset_name == 'train_c10n' and FLAGS.noisy_labels and FLAGS.eval_only:
        rand1, rand2, rand3 = inputs['label_rand1'], inputs['label_rand2'], inputs['label_rand3']
        nll_vec = (tf.keras.losses.sparse_categorical_crossentropy(rand1, probs) +
                   tf.keras.losses.sparse_categorical_crossentropy(rand2, probs) +
                   tf.keras.losses.sparse_categorical_crossentropy(rand3, probs))
        negative_log_likelihood = tf.reduce_mean(nll_vec) / 3.0
        metrics[f'{dataset_split}/negative_log_likelihood_c10n'].update_state(
          negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy_nl'].update_state(labels_noisy, probs)
      elif dataset_name.startswith('ood'):
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
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)

    if not FLAGS.eval_only:
      train_start_time = time.time()
      train_step(train_iterator)
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

    if validation_dataset:
      validation_iterator = iter(validation_dataset)
      test_step(
          validation_iterator, 'validation', 'clean', steps_per_validation)
    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
        datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      logging.info('Starting to run eval at epoch: %s', epoch)
      test_start_time = time.time()
      test_step(test_iterator, 'test', dataset_name, steps_per_eval)
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
                  steps_per_ood[dataset_name])

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
