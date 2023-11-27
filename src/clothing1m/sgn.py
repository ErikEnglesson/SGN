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

""" Code to train the Shifted Gaussian Noise (SGN) method with a ResNet-50 on the Clothing1M dataset.
    Most relevant parts of the code are:
    - Implementations of log-ratio transforms, starting on line 75.
    - The function to create a shifted Gaussian distribution on line 105.
    - The training step function on line 330.

    The code uses and is based on the Uncertainty Baselines GitHub repo:
    https://github.com/google/uncertainty-baselines
"""

import os
import time
import atexit
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import uncertainty_baselines as ub
import utils  # local file import
from clothing1m_loader import Clothing1MGenerator # local file import
tfd = tfp.distributions
tfb = tfp.bijectors


from tensorboard.plugins.hparams import api as hp

flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')

# SGN flags
flags.DEFINE_float('label_smoothing', 0.1, 'Label smoothing parameter in (0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls > 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in (0, 1].')
flags.DEFINE_float(
    'alpha', 0.7, 'Exponential moving average weight for label estimation.')
flags.DEFINE_float(
    'beta', 0.9999, 'Exponential moving average weight for parameters of delta model.')


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

    assert labels is not None
    assert exponent is not None
    mu_ema = tf.stop_gradient(mu_ema)
    factor = 1.0-tf.math.pow(FLAGS.alpha, exponent)

    t = ilr_inv(labels)
    mean += factor * (t - mu_ema)

    return tfd.MultivariateNormalDiagPlusLowRank(loc=mean,
                                                  scale_diag=diag,
                                                  scale_perturb_factor=r,
                                                  validate_args=False,
                                                  allow_nan_stats=False)


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
  
  ## data augmentations
  std = tf.constant([0.3113, 0.3192, 0.3214])
  transforms = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomCrop(224,224),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.Normalization(mean=(0.6959, 0.6537, 0.6371), variance=std**2),
  ])

  # Note that stateless_{fold_in,split} may incur a performance cost, but a
  # quick side-by-side test seemed to imply this was minimal.
  seeds = tf.random.experimental.stateless_split(
      [FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  
  # Create clean dataset
  train_builder = Clothing1MGenerator(
    root = data_dir,
    type = 'train',
    seed=seeds[0],
    transforms=transforms,
    sample_balanced_train=True)
  
  train_dataset = train_builder.load(batch_size=batch_size,prefetch=16)
  train_iterator = next(iter(train_dataset)) # to set num_examples property
  train_dataset = strategy.experimental_distribute_dataset(
        train_dataset)
        
  validation_builder = Clothing1MGenerator(
      root = data_dir,
      type = 'val',
      seed=seeds[0],
      transforms=transforms)
      
  validation_batch_size = 4*batch_size
  validation_dataset = validation_builder.load(batch_size=validation_batch_size,prefetch=16)
  validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
  steps_per_validation = validation_builder.num_examples // validation_batch_size
  
  clean_test_builder = Clothing1MGenerator(
    root = data_dir,
    type = 'test',
    seed=seeds[0],
    transforms=transforms)


  clean_test_dataset = clean_test_builder.load(batch_size=validation_batch_size,prefetch=16)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }

  num_classes = train_builder.num_classes 
  dim_logits = get_dim_logits(num_classes)
  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_builder.num_examples // validation_batch_size
  
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', train_builder.num_examples)
  
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    x = tf.keras.applications.ResNet50(include_top=False,
                                     input_shape=(224,224,3),
                                     pooling='avg',
                                     weights='imagenet')(inputs)
    x = tf.keras.layers.Flatten()(x)
    mu = tf.keras.layers.Dense(dim_logits,activation='linear')(x)
    sigma = tf.keras.layers.Dense(dim_logits,activation='linear')(x) 
    model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])

    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.beta)

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    x = tf.keras.applications.ResNet50(include_top=False,
                                    input_shape=(224,224,3),
                                    pooling='avg',
                                    weights='imagenet')(inputs)
    x = tf.keras.layers.Flatten()(x)
    mu = tf.keras.layers.Dense(dim_logits,activation='linear')(x)
    sigma = tf.keras.layers.Dense(dim_logits,activation='linear')(x) 
    model_delta = tf.keras.Model(inputs=inputs, outputs=[mu, sigma])
    ema.apply(model.trainable_variables)


    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    # Linearly scale learning rate and the decay epochs by vanilla settings.
    logging.info('base learning rate of: %f', FLAGS.base_learning_rate)
    logging.info('learning rate decay ratio: %f', FLAGS.lr_decay_ratio)
    logging.info('warmup epochs: %f', FLAGS.lr_warmup_epochs)
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        FLAGS.base_learning_rate,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=FLAGS.lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=False)
    metrics = {
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
        'train/accuracy_delta':
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

    if FLAGS.saved_model_dir:
      logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)

  @tf.function
  def train_step(iterator, epoch):
    """Training StepFn."""
    def step_fn(inputs, step):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      smoothed_targets = get_smoothed_onehot(labels, num_classes)
      logit_targets = ilr_inv(smoothed_targets)

      loc_ema, _ = model_delta(images, training=True)

      exponent = tf.cast(epoch, tf.float32) + \
                tf.cast(step, tf.float32) / steps_per_epoch


      with tf.GradientTape() as tape:
        loc, scale = model(images, training=True)
        
        normal = _create_normal(loc, scale, loc_ema, num_classes, smoothed_targets, exponent)

        nll_vec = normal.log_prob(logit_targets)
        negative_log_likelihood = -tf.reduce_mean(nll_vec)
       
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the weights. This excludes BN parameters and biases, but
          # pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
            
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss
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
      probs_delta = ilr_forward(loc_ema, axis=1)
      metrics['train/accuracy_delta'].update_state(labels, probs_delta)

    for step in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator), step,))
      
  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps, model_eval, suffix):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      
      images = inputs['features']
      labels = inputs['labels']
      loc, _ = model_eval(images, training=False)
      probs = ilr_forward(loc)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        if dataset_split == 'validation':
          metrics[f'{dataset_split}/negative_log_likelihood' + suffix].update_state(
            negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy_tl' + suffix].update_state(labels, probs)
          metrics[f'{dataset_split}/ece' + suffix].add_batch(probs, label=labels)
        else:
          metrics[f'{dataset_split}/negative_log_likelihood' + suffix].update_state(
              negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy' + suffix].update_state(labels, probs)
          metrics[f'{dataset_split}/ece' + suffix].add_batch(probs, label=labels)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})

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
    train_iterator = iter(train_dataset)
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    
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
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)
    
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    if validation_dataset:
      logging.info('Validating on clothing1m')
      for model_eval, suffix in zip(models, suffixes):
        validation_iterator = iter(validation_dataset)
        test_step(
          validation_iterator, 'validation', 'clean', steps_per_validation,
          model_eval, suffix)
      
    datasets_to_evaluate = {'clean': test_datasets['clean']}

    for dataset_name, test_dataset in datasets_to_evaluate.items():
      logging.info('Testing on dataset %s', dataset_name)
      logging.info('Starting to run eval at epoch: %s', epoch)
      test_start_time = time.time()
      for model_eval, suffix in zip(models, suffixes):
        test_iterator = iter(test_dataset)
        test_step(test_iterator, 'test', dataset_name, steps_per_eval, model_eval, suffix)

      ms_per_example = (time.time() - test_start_time) * 1e6 / validation_batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    
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

  final_checkpoint_delta_name = checkpoint_delta.save(
      os.path.join(FLAGS.output_dir, 'checkpoint_delta'))


  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  logging.info('Saved last delta checkpoint to %s', final_checkpoint_delta_name)


  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
    })
  atexit.register(strategy._extended._collective_ops._pool.close)

if __name__ == '__main__':
  app.run(main)
