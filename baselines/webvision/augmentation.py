# Hastily copied from:
# https://github.com/tensorflow/models/blob/v2.14.2/official/vision/ops/augment.py#L2619-L2780
# during the rebuttal.

import inspect
import math
from typing import Any, List, Iterable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
  """Fills blank area."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image

class MixupAndCutmix:
  """Applies Mixup and/or Cutmix to a batch of images.

  - Mixup: https://arxiv.org/abs/1710.09412
  - Cutmix: https://arxiv.org/abs/1905.04899

  Implementaion is inspired by https://github.com/rwightman/pytorch-image-models
  """

  def __init__(self,
               mixup_alpha: float = .8,
               cutmix_alpha: float = 1.,
               prob: float = 1.0,
               switch_prob: float = 0.5,
               label_smoothing: float = 0.1,
               num_classes: int = 1001):
    """Applies Mixup and/or Cutmix to a batch of images.

    Args:
      mixup_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Mixup is deactivated.
        Defaults to .8.
      cutmix_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Cutmix is deactivated.
        Defaults to 1..
      prob (float, optional): Of augmenting the batch. Defaults to 1.0.
      switch_prob (float, optional): Probability of applying Cutmix for the
        batch. Defaults to 0.5.
      label_smoothing (float, optional): Constant for label smoothing. Defaults
        to 0.1.
      num_classes (int, optional): Number of classes. Defaults to 1001.
    """
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.mix_prob = prob
    self.switch_prob = switch_prob
    self.label_smoothing = label_smoothing
    self.num_classes = num_classes
    self.mode = 'batch'
    self.mixup_enabled = True

    if self.mixup_alpha and not self.cutmix_alpha:
      self.switch_prob = -1
    elif not self.mixup_alpha and self.cutmix_alpha:
      self.switch_prob = 1

  def __call__(self, images: tf.Tensor,
               labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return self.distort(images, labels)

  def distort(self, images: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Mixup and/or Cutmix to batch of images and transforms labels.

    Args:
      images (tf.Tensor): Of shape [batch_size, height, width, 3] representing a
        batch of image, or [batch_size, time, height, width, 3] representing a
        batch of video.
      labels (tf.Tensor): Of shape [batch_size, ] representing the class id for
        each image of the batch.

    Returns:
      Tuple[tf.Tensor, tf.Tensor]: The augmented version of `image` and
        `labels`.
    """
    labels = tf.reshape(labels, [-1])
    augment_cond = tf.less(
        tf.random.uniform(shape=[], minval=0., maxval=1.0), self.mix_prob)
    # pylint: disable=g-long-lambda
    augment_a = lambda: self._update_labels(*tf.cond(
        tf.less(
            tf.random.uniform(shape=[], minval=0., maxval=1.0), self.switch_prob
        ), lambda: self._cutmix(images, labels), lambda: self._mixup(
            images, labels)))
    augment_b = lambda: (images, self._smooth_labels(labels))
    # pylint: enable=g-long-lambda

    return tf.cond(augment_cond, augment_a, augment_b)

  @staticmethod
  def _sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1., beta=alpha)
    sample_beta = tf.random.gamma(shape, 1., beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

  def _cutmix(self, images: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies cutmix."""
    lam = MixupAndCutmix._sample_from_beta(self.cutmix_alpha, self.cutmix_alpha,
                                           tf.shape(labels))

    ratio = tf.math.sqrt(1 - lam)

    batch_size = tf.shape(images)[0]

    if images.shape.rank == 4:
      image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
      fill_fn = _fill_rectangle
    elif images.shape.rank == 5:
      image_height, image_width = tf.shape(images)[2], tf.shape(images)[3]
      fill_fn = _fill_rectangle_video
    else:
      raise ValueError('Bad image rank: {}'.format(images.shape.rank))

    cut_height = tf.cast(
        ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)
    cut_width = tf.cast(
        ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)

    random_center_height = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32)
    random_center_width = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32)

    bbox_area = cut_height * cut_width
    lam = 1. - bbox_area / (image_height * image_width)
    lam = tf.cast(lam, dtype=tf.float32)

    images = tf.map_fn(
        lambda x: fill_fn(*x),
        (images, random_center_width, random_center_height, cut_width // 2,
         cut_height // 2, tf.reverse(images, [0])),
        dtype=(
            images.dtype, tf.int32, tf.int32, tf.int32, tf.int32, images.dtype),
        fn_output_signature=tf.TensorSpec(images.shape[1:], dtype=images.dtype))

    return images, labels, lam

  def _mixup(self, images: tf.Tensor,
             labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Applies mixup."""
    lam = MixupAndCutmix._sample_from_beta(self.mixup_alpha, self.mixup_alpha,
                                           tf.shape(labels))
    if images.shape.rank == 4:
      lam = tf.reshape(lam, [-1, 1, 1, 1])
    elif images.shape.rank == 5:
      lam = tf.reshape(lam, [-1, 1, 1, 1, 1])
    else:
      raise ValueError('Bad image rank: {}'.format(images.shape.rank))

    lam_cast = tf.cast(lam, dtype=images.dtype)
    images = lam_cast * images + (1. - lam_cast) * tf.reverse(images, [0])

    return images, labels, tf.squeeze(lam)

  def _smooth_labels(self, labels: tf.Tensor) -> tf.Tensor:
    off_value = self.label_smoothing / self.num_classes
    on_value = 1. - self.label_smoothing + off_value

    smooth_labels = tf.one_hot(
        labels, self.num_classes, on_value=on_value, off_value=off_value)
    return smooth_labels

  def _update_labels(self, images: tf.Tensor, labels: tf.Tensor,
                     lam: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    labels_1 = self._smooth_labels(labels)
    labels_2 = tf.reverse(labels_1, [0])

    lam = tf.reshape(lam, [-1, 1])
    labels = lam * labels_1 + (1. - lam) * labels_2

    return images, labels
