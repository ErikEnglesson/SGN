"""WebVision dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

from PIL import Image

# TODO(WebVision): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(WebVision): BibTeX citation
_CITATION = """
"""


class WebVision(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for WebVision dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(WebVision): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(224, 224, 3)),
            'label': tfds.features.ClassLabel(num_classes=50),
            'index': tfds.features.Tensor(shape=(), dtype=tf.int32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    self.root = '/scratch/local/WebVision/'
    self.split_path = {'train': '',
                       'val': 'val_images_256/'}
    self.image_paths_files = {'train': self.root + 'train_filelist_google.txt',
                              'val': self.root + 'val_filelist.txt'}


    # Number of samples for different dataset types
    self.num_samples = {'train': 65944,
                        'val': 2500}
    self.num_classes = 50 


    # TODO(WebVision): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples('train'),
        'val': self._generate_examples('val')
    }

  def _generate_examples(self, split):
    """Yields examples."""

    img_paths, path_to_label = self._load_paths_and_labels(split)
    assert self.num_samples[split] == len(img_paths)
    offset = 0 if split == 'train' else self.num_samples['train']
    for i, path in enumerate(img_paths):
      index = i + offset
      yield str(index), {
              'image': self.root + self.split_path[split] + path,
              'label': path_to_label[path],
              'index': index,
      }


  def _load_paths_and_labels(self, split):
    path = self.image_paths_files[split]
    image_paths = []
    paths_to_labels = dict()
    with open(path, 'r') as file:
      for i, line in enumerate(file):
          path, label = line.split()

          if int(label) >= self.num_classes:
              continue

          paths_to_labels[path] = int(label)
          image_paths.append(path)

    return image_paths, paths_to_labels

