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

"""Wide ResNet 28-10 on CIFAR-10/100 trained with maximum likelihood.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.
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
import uncertainty_baselines as ub
import ood_utils  # local file import
import utils  # local file import
# local file import
from label_corrupted_dataset import make_label_corrupted_dataset, get_element_id


from tensorboard.plugins.hparams import api as hp


flags.DEFINE_float('label_smoothing', 0.,
                   'Label smoothing parameter in [0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls >= 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in [0, 1].')

# Data Augmentation flags.
flags.DEFINE_bool('augmix', False,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
                     'Number of augmentation operations in AugMix to perform '
                     'on the input image. In the simgle model context, it'
                     'should be 1. In the ensembles context, it should be'
                     'ensemble_size if we perform random_augment only; It'
                     'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth. This corresponds'
                     'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
                     'Augmix width. This corresponds to the k in line 5 in the'
                     'Algorithm box in [4].')

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
flags.DEFINE_float('beta', 0.7,
                   'early learning regularization beta.'
                   ' for more information refer to '
                   'https://github.com/shengliu66/ELR'
                   'default value is taken from'
                   'https://github.com/shengliu66/ELR/blob/master/ELR/config_cifar10.json')

flags.DEFINE_float('reg_scale', 3.0,
                   'early learning regularization lambda.'
                   ' for more information refer to '
                   'https://github.com/shengliu66/ELR'
                   'default value is taken from'
                   'https://github.com/shengliu66/ELR/blob/master/ELR/config_cifar10.json')


flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')

# OOD flags.
flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', False,
                  'Whether to run OOD evaluation on specified OOD datasets.')
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
                  enum_values=['sym', 'asym', 'aggre', 'worst',
                               'rand1', 'rand2', 'rand3', 'c100noise'],
                  help='Type of label noise.')


FLAGS = flags.FLAGS


def _extract_hyperparameter_dictionary():
    """Create the dictionary of hyperparameters from FLAGS."""
    flags_as_dict = FLAGS.flag_values_dict()
    hp_keys = ub.models.models.wide_resnet.HP_KEYS
    hps = {k: flags_as_dict[k] for k in hp_keys}
    return hps


def _generalized_energy_distance(labels, predictions, num_classes):
    """Compute generalized energy distance.

    See Eq. (8) https://arxiv.org/abs/2006.06015
    where d(a, b) = (a - b)^2.

    Args:
      labels: [batch_size, num_classes] Tensor with empirical probabilities of
        each class assigned by the labellers.
      predictions: [batch_size, num_classes] Tensor of predicted probabilities.
      num_classes: Integer.

    Returns:
      Tuple of Tensors (label_diversity, sample_diversity, ged).
    """
    y = tf.expand_dims(labels, -1)
    y_hat = tf.expand_dims(predictions, -1)

    non_diag = tf.expand_dims(1.0 - tf.eye(num_classes), 0)
    distance = tf.reduce_sum(tf.reduce_sum(
        non_diag * y * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
    label_diversity = tf.reduce_sum(tf.reduce_sum(
        non_diag * y * tf.transpose(y, perm=[0, 2, 1]), -1), -1)
    sample_diversity = tf.reduce_sum(tf.reduce_sum(
        non_diag * y_hat * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
    ged = tf.reduce_mean(2 * distance - label_diversity - sample_diversity)
    return label_diversity, sample_diversity, ged


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
    logging.info('Size of the dataset %s',
                 ds_info.splits['train'].num_examples)
    logging.info('Train proportion %s', FLAGS.train_proportion)
    steps_per_eval = ds_info.splits['test'].num_examples // batch_size
    num_classes = ds_info.features['label'].num_classes

    aug_params = {
        'augmix': FLAGS.augmix,
        'aug_count': FLAGS.aug_count,
        'augmix_depth': FLAGS.augmix_depth,
        'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
        'augmix_width': FLAGS.augmix_width,
    }

    # Note that stateless_{fold_in,split} may incur a performance cost, but a
    # quick side-by-side test seemed to imply this was minimal.
    seeds = tf.random.experimental.stateless_split(
        [FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]

    # Create clean dataset
    train_builder = ub.datasets.get(
        FLAGS.dataset,
        data_dir=data_dir,
        download_data=FLAGS.download_data,
        split=tfds.Split.TRAIN,
        seed=seeds[0],
        aug_params=aug_params,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        validation_percent=1. - FLAGS.train_proportion,
    )

    if FLAGS.noisy_labels:
        # Create noisy dataset by augmenting clean with extra noisy label
        dataset_cls = ub.datasets.DATASETS[FLAGS.dataset]
        dataset_cls = make_label_corrupted_dataset(dataset_cls)
        train_builder = dataset_cls(
            dataset=train_builder,
            split=tfds.Split.TRAIN,
            data_dir=data_dir,
            seed=seeds[0],
            aug_params=aug_params,
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
            split=tfds.Split.VALIDATION,
            validation_percent=1. - FLAGS.train_proportion,
            data_dir=data_dir)
        if FLAGS.noisy_labels:
            dataset_cls = ub.datasets.DATASETS[FLAGS.dataset]
            dataset_cls = make_label_corrupted_dataset(dataset_cls)
            validation_builder = dataset_cls(
                dataset=validation_builder,
                split=tfds.Split.VALIDATION,
                data_dir=data_dir,
                seed=seeds[0],
                validation_percent=1.0 - FLAGS.train_proportion,
                corruption_type=FLAGS.corruption_type,
                severity=FLAGS.severity)

        validation_dataset = validation_builder.load(batch_size=batch_size)
        validation_dataset = strategy.experimental_distribute_dataset(
            validation_dataset)
        steps_per_validation = validation_builder.num_examples // batch_size
    clean_test_builder = ub.datasets.get(
        FLAGS.dataset,
        split=tfds.Split.TEST,
        data_dir=data_dir)
    clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
    test_datasets = {
        'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
    }

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    steps_per_epoch = train_builder.num_examples // batch_size
    steps_per_eval = clean_test_builder.num_examples // batch_size
    num_classes = 100 if FLAGS.dataset == 'cifar100' else 10

    if FLAGS.eval_on_ood:
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
                    severity=severity,
                    split=tfds.Split.TEST,
                    data_dir=data_dir).load(batch_size=batch_size)
                test_datasets[f'{corruption_type}_{severity}'] = (
                    strategy.experimental_distribute_dataset(dataset))

    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.output_dir, 'summaries'))

    with strategy.scope():
        logging.info('Building ResNet model')
        model = ub.models.wide_resnet(
            input_shape=(32, 32, 3),
            depth=28,
            width_multiplier=FLAGS.width,
            num_classes=num_classes,
            l2=FLAGS.l2,
            hps=_extract_hyperparameter_dictionary(),
            seed=seeds[1])
        logging.info('Model input shape: %s', model.input_shape)
        logging.info('Model output shape: %s', model.output_shape)
        logging.info('Model number of weights: %s', model.count_params())
        # Linearly scale learning rate and the decay epochs by vanilla settings.

        base_lr = FLAGS.base_learning_rate * batch_size / 128
        optimizer = tf.keras.optimizers.SGD(base_lr,
                                            momentum=1.0 - FLAGS.one_minus_momentum,
                                            nesterov=True)
       
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
        if FLAGS.eval_on_ood:
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

        if FLAGS.saved_model_dir:
            logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
            latest_checkpoint = tf.train.latest_checkpoint(
                FLAGS.saved_model_dir)
            checkpoint.restore(latest_checkpoint)
            logging.info('Loaded checkpoint %s', latest_checkpoint)
        if FLAGS.eval_only:
            initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval

    @tf.function
    def train_step(iterator, preds_buffer):
        """Training StepFn."""
        def step_fn(inputs, preds_buffer):
            """Per-Replica StepFn."""

            # Extract images, labels, and indices for the examples in the batch.
            images = inputs['features']
            labels = inputs['noisy_labels'] if FLAGS.noisy_labels else inputs['labels']
            indices = tf.map_fn(
                get_element_id, inputs['id'], fn_output_signature=tf.int64)

            if FLAGS.augmix and FLAGS.aug_count >= 1:
                # Index 0 at augmix processing is the unperturbed image.
                # We take just 1 augmented image from the returned augmented images.
                images = images[:, 1, ...]

    
            with tf.GradientTape() as tape:
                # -----------------------------------------------------------------------------
                # A re-implementation of ELR in TensorFlow directly based on the official code:
                # https://github.com/shengliu66/ELR/blob/master/ELR/model/loss.py#L21
                # -----------------------------------------------------------------------------
                logits = model(images, training=True)
                preds = tf.nn.softmax(logits, axis=1)
                preds = tf.clip_by_value(preds, 1e-4, 1-1e-4)
                preds_ = tf.stop_gradient(preds)
                preds_buffer_update = tf.gather(preds_buffer, indices)
                preds_buffer_update = FLAGS.beta*preds_buffer_update + \
                    (1-FLAGS.beta) * \
                    (preds_/tf.reduce_sum(preds_, axis=1, keepdims=True))
                indices = tf.expand_dims(indices, axis=1)
                preds_buffer = tf.tensor_scatter_nd_update(
                    preds_buffer, indices, preds_buffer_update)
                max_indices = tf.reduce_max(indices)

                elr_loss = tf.reduce_mean(tf.math.log(
                    1-tf.reduce_sum(preds_buffer_update * preds, axis=1)))

                if FLAGS.label_smoothing == 0.:
                    negative_log_likelihood = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                        logits,
                                                                        from_logits=True))
                else:
                    one_hot_labels = tf.one_hot(
                        tf.cast(labels, tf.int32), num_classes)
                    negative_log_likelihood = tf.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(
                            one_hot_labels,
                            logits,
                            from_logits=True,
                            label_smoothing=FLAGS.label_smoothing))
                l2_loss = sum(model.losses)
                loss = negative_log_likelihood + l2_loss + FLAGS.reg_scale * elr_loss
                # Scale the loss given the TPUStrategy will reduce sum all gradients.
                scaled_loss = loss / strategy.num_replicas_in_sync

            grads = tape.gradient(scaled_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            probs = tf.nn.softmax(logits)
            metrics['train/ece'].add_batch(probs, label=labels)
            metrics['train/loss'].update_state(loss)
            metrics['train/negative_log_likelihood'].update_state(
                negative_log_likelihood)
            metrics['train/accuracy'].update_state(labels, logits)

            if FLAGS.noisy_labels:
                mask = inputs['labels'] == inputs['noisy_labels']
                tl_labels_clean, tl_labels_corrupted = tf.boolean_mask(
                    inputs['labels'], mask), tf.boolean_mask(inputs['labels'], ~mask)
                nl_labels_corrupted = tf.boolean_mask(
                    inputs['noisy_labels'], ~mask)
                probs_clean, probs_corrupted = tf.boolean_mask(
                    probs, mask), tf.boolean_mask(probs, ~mask)

                # Accuracy
                metrics['train/accuracy_tl_clean'].update_state(
                    tl_labels_clean, probs_clean)
                metrics['train/accuracy_tl_corrupted'].update_state(
                    tl_labels_corrupted, probs_corrupted)
                metrics['train/accuracy_nl_corrupted'].update_state(
                    nl_labels_corrupted, probs_corrupted)
            return preds_buffer, max_indices

        max_indices = tf.constant(-1, dtype=tf.int64)
        step_max_indices = tf.constant(-1, dtype=tf.int64)
        for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
            preds_buffer, step_max_indices = strategy.run(
                step_fn, args=(next(iterator), preds_buffer))
            max_indices = tf.maximum(max_indices, step_max_indices)
        return preds_buffer, max_indices

    @tf.function
    def test_step(iterator, dataset_split, dataset_name, num_steps):
        """Evaluation StepFn."""
        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs['features']
            labels = inputs['labels']
            labels_noisy = inputs['noisy_labels'] if FLAGS.noisy_labels and dataset_split == 'validation' else inputs['labels']
            logits = model(images, training=False)
            probs = tf.nn.softmax(logits)

            negative_log_likelihood = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

            if dataset_name == 'clean':
                if dataset_split == 'validation':
                    metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
                        negative_log_likelihood)
                    metrics[f'{dataset_split}/accuracy_tl'].update_state(
                        labels, probs)
                    metrics[f'{dataset_split}/accuracy_nl'].update_state(
                        labels_noisy, probs)
                    metrics[f'{dataset_split}/ece'].add_batch(
                        probs, label=labels)
                else:
                    metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
                        negative_log_likelihood)
                    metrics[f'{dataset_split}/accuracy'].update_state(
                        labels, probs)
                    metrics[f'{dataset_split}/ece'].add_batch(
                        probs, label=labels)
            elif dataset_name.startswith('ood'):
                ood_labels = 1 - inputs['is_in_distribution']
                if FLAGS.dempster_shafer_ood:
                    ood_scores = ood_utils.DempsterShaferUncertainty(logits)
                else:
                    ood_scores = 1 - tf.reduce_max(probs, axis=-1)

                # Edgecase for if dataset_name contains underscores
                ood_dataset_name = '_'.join(dataset_name.split('_')[1:])
                for name, metric in metrics.items():
                    if ood_dataset_name in name:
                        metric.update_state(ood_labels, ood_scores)
            else:
                corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
                    negative_log_likelihood)
                corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
                    labels, probs)
                corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
                    probs, label=labels)

        for _ in tf.range(tf.cast(num_steps, tf.int32)):
            strategy.run(step_fn, args=(next(iterator),))

    @tf.function
    def cifar10h_test_step(iterator, num_steps):
        """Evaluation StepFn."""
        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs['features']
            labels = inputs['labels']
            logits = model(images, training=False)

            negative_log_likelihood = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)(labels, logits)

            negative_log_likelihood = tf.reduce_mean(negative_log_likelihood)
            metrics['cifar10h/nll'].update_state(negative_log_likelihood)

            label_diversity, sample_diversity, ged = _generalized_energy_distance(
                labels, tf.nn.softmax(logits), 10)

            metrics['cifar10h/ged'].update_state(ged)
            metrics['cifar10h/ged_label_diversity'].update_state(
                tf.reduce_mean(label_diversity))
            metrics['cifar10h/ged_sample_diversity'].update_state(
                tf.reduce_mean(sample_diversity))

        for _ in tf.range(tf.cast(num_steps, tf.int32)):
            strategy.run(step_fn, args=(next(iterator),))

    metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
    metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})

    train_iterator = iter(train_dataset)
    # ? shape and init location unknown
    preds_buffer = tf.zeros(shape=(50000, num_classes))
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
            preds_buffer, max_indices = train_step(
                train_iterator, preds_buffer)
            assert max_indices < preds_buffer.shape[0], f"there is an error in indexing preds_buffer.shape[0] must be bigger "\
                f"than max_indices but {preds_buffer.shape[0]}<{max_indices}"


            ms_per_example = (
                time.time() - train_start_time) * 1e6 / batch_size
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

        if FLAGS.eval_on_ood:
            for dataset_name in ood_dataset_names:
                ood_iterator = iter(
                    ood_datasets['ood_{}'.format(dataset_name)])
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
        total_results = {name: metric.result()
                         for name, metric in metrics.items()}
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

        if FLAGS.corruptions_interval > 0:
            for metric in corrupt_metrics.values():
                metric.reset_states()

        if (FLAGS.checkpoint_interval > 0 and
                (epoch + 1) % FLAGS.checkpoint_interval == 0):
            checkpoint_name = checkpoint.save(
                os.path.join(FLAGS.output_dir, 'checkpoint'))
            logging.info('Saved checkpoint to %s', checkpoint_name)

    final_checkpoint_name = checkpoint.save(
        os.path.join(FLAGS.output_dir, 'checkpoint'))
    logging.info('Saved last checkpoint to %s', final_checkpoint_name)
    with summary_writer.as_default():
        hp.hparams({
            'base_learning_rate': FLAGS.base_learning_rate,
            'one_minus_momentum': FLAGS.one_minus_momentum,
            'l2': FLAGS.l2,
        })
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    app.run(main)
