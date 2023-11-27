import logging
from typing import Callable, Dict, Optional, Sequence, Type, TypeVar, Union

from robustness_metrics.common import ops
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf

import functools

from tensorflow.python.ops import stateless_random_ops
from tensorflow_datasets.core import dataset_utils
import numpy as np
from scipy.special import softmax
import uncertainty_baselines as ub


# For datasets like UCI, the tf.data.Dataset returned by _read_examples will
# have elements that are Sequence[tf.Tensor], for TFDS datasets they will be
# Dict[Text, tf.Tensor] (types.Features), for Criteo they are a tf.Tensor.
PreProcessFn = Callable[
    [Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]],
    types.Features]

_BaseDatasetClass = Type[TypeVar('B', bound=ub.datasets.BaseDataset)]


def get_element_id(example_id):
    tf.debugging.Assert(tf.strings.substr(example_id, 0, 6)
                        == 'train_', [example_id])
    return tf.strings.to_number(tf.strings.substr(example_id, 6, -1), out_type=tf.int64)

# This implementation is adapted to TensorFlow from:
# https://github.com/haochenglouis/cores/blob/main/data/utils.py#L185


def noisify_instance(train_data, train_labels, noise_rate):
    if max(train_labels) > 10:
        num_class = 100
    else:
        num_class = 10

    # We do not set the seed here as we want to change it for different runs.
    # np.random.seed(0)
    q_ = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q) == 50000:
            break

    w = np.random.normal(loc=0, scale=1, size=(32*32*3, num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample, w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i] * softmax(p_all, axis=0)
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(
            np.arange(num_class), p=p_all/sum(p_all)))

    noisy_labels = np.array(noisy_labels).astype(np.int32)
    over_all_noise_rate = 1 - \
        float((train_labels == noisy_labels).sum())/train_labels.shape[0]
    return noisy_labels, over_all_noise_rate


def make_label_corrupted_dataset(dataset_cls: _BaseDatasetClass) -> _BaseDatasetClass:
    """Generate a BaseDataset with noisy labels."""

    class _LabelCorruptedBaseDataset(dataset_cls):
        """Add noisy labels."""

        def __init__(
                self,
                dataset: _BaseDatasetClass,
                severity: float = 0.4,
                corruption_type: str = 'asym',
                **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset
            self.severity = severity
            self.corruption_type = corruption_type
            corruptions_real = ['aggre', 'worst',
                                'rand1', 'rand2', 'rand3', 'c100noise']
            corruptions_synthetic = ['sym', 'asym', 'instance']
            assert corruption_type in corruptions_real + corruptions_synthetic
            self.is_synthetic = corruption_type in corruptions_synthetic
            is_c10 = dataset.name == 'cifar10'
            self.num_classes = 10 if is_c10 else 100

            # assert not (not is_c10 and self.is_synthetic)
            if not is_c10:
                assert corruption_type in [
                    'c100noise', 'sym', 'asym', 'instance']

            self.index_map = None
            self.clean_labels = None
            if is_c10 or corruption_type == 'c100noise':
                paths_labels = './src/cifar/datasets/CIFAR-10_human_ordered.npy' if is_c10 else './src/cifar/datasets/CIFAR-100_human_ordered.npy'
                paths_indices = './src/cifar/datasets/image_order_c10_inverted.npy' if is_c10 else './src/cifar/datasets/image_order_c100_inverted.npy'
                noise_file = np.load(paths_labels, allow_pickle=True)

                # Clean labels are used for sanity checks.
                self.clean_labels = tf.convert_to_tensor(
                    noise_file.item().get('clean_label'))

                # === How to convert from their indices to a useful map. ===
                # torch_to_tf = np.load('./datasets/image_order_c100.npy', allow_pickle=True)
                # tf_to_torch = np.zeros(len(torch_to_tf), dtype=np.int64)
                # for v in range(len(torch_to_tf)):
                #    tf_to_torch[torch_to_tf[v]] = np.where(torch_to_tf==torch_to_tf[v])[0][0]
                # np.save('./datasets/image_order_c100_inverted.npy', tf_to_torch)
                # tf_to_torch_loaded = np.load('./datasets/image_order_c100_inverted.npy')
                # assert np.array_equal(tf_to_torch, tf_to_torch_loaded)

                tf_to_torch = np.load(paths_indices)
                self.index_map = tf.convert_to_tensor(tf_to_torch)

                corruption_type_to_noise_key = {'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                                                'rand2': 'random_label2', 'rand3': 'random_label3', 'c100noise': 'noise_label'}

            self.extra_labels = None
            if is_c10:
                rand1 = tf.cast(tf.convert_to_tensor(
                    noise_file.item().get('random_label1')), tf.float32)
                rand2 = tf.cast(tf.convert_to_tensor(
                    noise_file.item().get('random_label2')), tf.float32)
                rand3 = tf.cast(tf.convert_to_tensor(
                    noise_file.item().get('random_label3')), tf.float32)
                self.extra_labels = {'rand1': rand1,
                                     'rand2': rand2, 'rand3': rand3}

            if corruption_type in corruptions_real:
                noise_key = corruption_type_to_noise_key[corruption_type]

                self.noisy_labels = tf.convert_to_tensor(
                    noise_file.item().get(noise_key))
            elif corruption_type == 'instance':
                # This is a hack to get the training loader to not repeat the dataset
                drop = dataset._drop_remainder
                is_training = dataset._is_training
                dataset._drop_remainder = False
                dataset._is_training = False
                ds = dataset.load(batch_size=128)
                dataset._drop_remainder = drop
                dataset._is_training = is_training

                train_data, train_labels, element_ids = [], [], []
                for i, ex in enumerate(ds.as_numpy_iterator()):
                    train_data.append(ex['features'])
                    train_labels.append(ex['labels'])
                    element_ids.extend([int(ex_id[6:]) for ex_id in ex['id']])

                train_data = np.concatenate(train_data)
                train_labels = np.concatenate(train_labels).astype(np.int32)
                element_ids = np.array(element_ids)
                id_to_index = np.zeros(50000, dtype=np.int64)
                for v in range(len(element_ids)):
                    id_to_index[element_ids[v]] = np.where(
                        element_ids == element_ids[v])[0][0]

                self.noisy_labels, _ = noisify_instance(
                    train_data, train_labels, severity)

                self.clean_labels = tf.convert_to_tensor(
                    train_labels.astype(np.float32))
                self.noisy_labels = tf.convert_to_tensor(
                    self.noisy_labels.astype(np.float32))
                self.index_map = tf.convert_to_tensor(id_to_index)
            else:
                self.noisy_labels = None

        def load(self,
                 *,
                 preprocess_fn=None,
                 batch_size: int = -1) -> tf.data.Dataset:
            if preprocess_fn:
                dataset_preprocess_fn = preprocess_fn
            else:
                dataset_preprocess_fn = (
                    self.dataset._create_process_example_fn())  # pylint: disable=protected-access

            noisy_label_fn = None
            if self.corruption_type == 'sym':
                noisy_label_fn = _create_uniform_noisy_label_fn
            elif self.corruption_type == 'asym':
                noisy_label_fn = _create_asym_noisy_label_fn
            elif self.corruption_type == 'instance':
                noisy_label_fn = _create_instance_noisy_label_fn
            else:
                assert not self.is_synthetic
                noisy_label_fn = _create_real_noisy_label_fn

            dataset_preprocess_fn = ops.compose(
                dataset_preprocess_fn,
                noisy_label_fn(self.num_classes, self._seed, self.severity, self.index_map, self.clean_labels, self.noisy_labels, self.extra_labels))
            dataset = self.dataset.load(
                preprocess_fn=dataset_preprocess_fn,
                batch_size=batch_size)

            return dataset

    return _LabelCorruptedBaseDataset


def _create_asym_noisy_label_fn(num_classes, seed, severity, id_map=None, clean_labels=None, noisy_labels=None, extra_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _flip_label_c10(example: types.Features) -> types.Features:
        i = int(example['labels'])
        if i == 9:
            return 1.0
        # bird -> airplane
        elif i == 2:
            return 0.0
        # cat -> dog
        elif i == 3:
            return 5.0
        # dog -> cat
        elif i == 5:
            return 3.0
        # deer -> horse
        elif i == 4:
            return 7.0
        else:
            return float(i)

    def _flip_label_c100(example: types.Features) -> types.Features:
        i = int(example['labels'])
        return float((i+1) % 100)

    def _add_noisy_label(example: types.Features) -> types.Features:
        per_example_seed = tf.random.experimental.stateless_fold_in(
            seed, example['element_id'][0])
        random_func = functools.partial(
            stateless_random_ops.stateless_random_uniform, seed=per_example_seed)
        uniform_random = random_func(shape=[], minval=0, maxval=1.0)
        flip_cond = tf.math.less(uniform_random, severity)
        example['noisy_labels'] = tf.cond(flip_cond, lambda: _flip_label_c10(
            example) if num_classes == 10 else _flip_label_c100(example), lambda: example['labels'])

        if num_classes == 10:
            assert id_map is not None

            element_id = example['id']
            tf.debugging.Assert(tf.strings.substr(
                element_id, 0, 6) == 'train_', [element_id])
            element_id = tf.strings.to_number(
                tf.strings.substr(element_id, 6, -1), out_type=tf.int64)
            mapped_id = id_map[element_id]

            example['label_rand1'] = extra_labels['rand1'][mapped_id]
            example['label_rand2'] = extra_labels['rand2'][mapped_id]
            example['label_rand3'] = extra_labels['rand3'][mapped_id]

        return example

    return _add_noisy_label


def _create_uniform_noisy_label_fn(num_classes, seed, severity, id_map=None, clean_labels=None, noisy_labels=None, extra_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _flip_label(example: types.Features, random_func) -> types.Features:
        return float(random_func(shape=[], minval=0, maxval=num_classes, dtype=tf.int32))

    def _add_noisy_label(example: types.Features) -> types.Features:
        per_example_seed = tf.random.experimental.stateless_fold_in(
            seed, example['element_id'][0])
        random_func = functools.partial(
            stateless_random_ops.stateless_random_uniform, seed=per_example_seed)
        uniform_random = random_func(shape=[], minval=0, maxval=1.0)
        flip_cond = tf.math.less(uniform_random, severity)
        example['noisy_labels'] = tf.cond(flip_cond, lambda: _flip_label(
            example, random_func), lambda: example['labels'])

        if num_classes == 10:
            assert id_map is not None
            element_id = example['id']
            tf.debugging.Assert(tf.strings.substr(
                element_id, 0, 6) == 'train_', [element_id])
            element_id = tf.strings.to_number(
                tf.strings.substr(element_id, 6, -1), out_type=tf.int64)
            mapped_id = id_map[element_id]

            example['label_rand1'] = extra_labels['rand1'][mapped_id]
            example['label_rand2'] = extra_labels['rand2'][mapped_id]
            example['label_rand3'] = extra_labels['rand3'][mapped_id]

        return example

    return _add_noisy_label


def _create_real_noisy_label_fn(num_classes, seed, severity, id_map, clean_labels, noisy_labels, extra_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _add_noisy_label(example: types.Features) -> types.Features:
        element_id = example['id']
        tf.debugging.Assert(tf.strings.substr(
            element_id, 0, 6) == 'train_', [element_id])
        element_id = tf.strings.to_number(
            tf.strings.substr(element_id, 6, -1), out_type=tf.int64)
        mapped_id = id_map[element_id]
        clean_label = tf.cast(clean_labels[mapped_id], dtype=tf.float32)
        noisy_label = tf.cast(noisy_labels[mapped_id], dtype=tf.float32)
        tf.debugging.Assert(example['labels'] == clean_label, [
                            'not equal', element_id, mapped_id, example['labels'], clean_label])

        example['noisy_labels'] = noisy_label

        if num_classes == 10:
            assert id_map is not None
            example['label_rand1'] = extra_labels['rand1'][mapped_id]
            example['label_rand2'] = extra_labels['rand2'][mapped_id]
            example['label_rand3'] = extra_labels['rand3'][mapped_id]

        return example

    return _add_noisy_label


def _create_instance_noisy_label_fn(num_classes, seed, severity, id_map=None, clean_labels=None, noisy_labels=None, extra_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _add_noisy_label(example: types.Features) -> types.Features:
        element_id = example['id']
        tf.debugging.Assert(tf.strings.substr(
            element_id, 0, 6) == 'train_', [element_id])
        element_id = tf.strings.to_number(
            tf.strings.substr(element_id, 6, -1), out_type=tf.int64)
        mapped_id = id_map[element_id]
        clean_label = tf.cast(clean_labels[mapped_id], dtype=tf.float32)
        noisy_label = tf.cast(noisy_labels[mapped_id], dtype=tf.float32)

        tf.debugging.assert_equal(example['labels'], clean_label)

        example['noisy_labels'] = noisy_label
        if num_classes == 10:
            assert id_map is not None
            example['label_rand1'] = extra_labels['rand1'][mapped_id]
            example['label_rand2'] = extra_labels['rand2'][mapped_id]
            example['label_rand3'] = extra_labels['rand3'][mapped_id]

        return example

    return _add_noisy_label
