import tensorflow as tf

import os
import random
import numpy as np
from PIL import Image

class Clothing1MGenerator:
    def __init__(self, 
                 root,
                 type='train', 
                 seed=0,
                 transforms=None, 
                 target_transform=None, 
                 sample_balanced_train=False,
                 shuffle_buffer_size = None,
                 trainset_size=265664):
        
        assert type == 'train' or type == 'val' or type == 'test'
        assert ~sample_balanced_train or type == 'train'
        
        self.type = type
        self.shuffle_buffer_size = shuffle_buffer_size
        self.root = root
        self.seed = tf.cast(seed,dtype=tf.int64)
        self.image_transforms = transforms
        self.target_transform = target_transform
        self.sample_balanced_train = sample_balanced_train
        
        self.image_paths_files = {'train': 'noisy_train_key_list.txt', 
                                  'val': 'clean_val_key_list.txt',
                                  'test': 'clean_test_key_list.txt'}
        
        self.paths_to_labels_files = {'train': 'noisy_label_kv.txt', 
                                      'val': 'clean_label_kv.txt',
                                      'test': 'clean_label_kv.txt'}
        
        self.num_samples = {'train': 1000000, 
                            'val': 14313,
                            'test': 10526}
        self.num_classes = 14
        if type=='train' and sample_balanced_train:
          self.num_samples[type] = trainset_size
        else:
          self.num_examples = self.num_samples[type]
          self.img_paths = self.load_image_paths()
          assert self.num_examples == len(self.img_paths)
        self.path_to_label = self.load_paths_to_labels()
        self.label_to_name = self.load_labels_to_names()
            
    def load(self, batch_size,prefetch):
        """
            a wrapper functions to make the data loader compatible with the uncertainty beselines package
        """
        return self.get_dataloader(batch_size,prefetch)
      
    def get_dataloader(self, batch_size,prefetch=10):
        """
            a function that returns a tf.data.Dataset of clothing 1M dataset indexed by a tf.data.Dataset.range.
            it prefetches @prefetch batches into memory default=5.
        """
            
        # assert output signature
        loader = tf.data.Dataset.from_generator(
            self,
            output_signature={
              'features':tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
              'labels':tf.TensorSpec(shape=(), dtype=tf.int32)})
        
        return loader.batch(batch_size).prefetch(prefetch)
        
    def __call__(self):
        """
            a function to make the class callable for tf.data.Dataset.from_generator which is used in get_dataloader function.
        """
        return self
    
    def __iter__(self):
        """
            a wrapper function for getting the iterator over tf.Dataset.range dataset.
        """
        
        if self.sample_balanced_train:
            self.img_paths = self.sample_balanced_train_dataset()
            
        if self.shuffle_buffer_size is None:
          self.shuffle_buffer_size = self.num_examples
        self.indices = tf.data.Dataset.range(self.num_examples).shuffle(
            self.shuffle_buffer_size,
            seed=self.seed,
            reshuffle_each_iteration=False)
        
        self.indices_iterator = iter(self.indices)
        return self
    
    def __next__(self):
        """
            a wrapper function for getting the next index from tf.Dataset.range dataset and loading images corresponding to the indices.
        """
        index = next(self.indices_iterator)
        return self.__getitem__(index)
    
    def __getitem__(self,index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path = self.img_paths[index]
        label = self.path_to_label[path] 
        img = self.load_img(path)

        image = tf.convert_to_tensor(
            np.array(img.copy()),
            dtype=tf.float32)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return {'features':image,
                'labels':label}
    
    def get_label_name(self, label):
        assert label < 14 and label >= 0
        return self.label_to_name[label]

    def load_img(self, img_path):
        path = os.path.join(self.root, 
                            img_path)
        return tf.keras.utils.load_img(path)

    def load_paths_to_labels(self):
        path = os.path.join(self.root, 
                            self.paths_to_labels_files[self.type])
        paths_to_labels = dict()
        with open(path, 'r') as file:
            for line in file:
                path, label = line.split()
                paths_to_labels[path] = int(label)

        return paths_to_labels

    def load_image_paths(self):
        path = os.path.join(self.root, 
                            self.image_paths_files[self.type])
        
        image_paths = []
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                image_paths.append(line.replace('\n',''))
        
        return image_paths
                            
    def sample_balanced_train_dataset(self):
        self.img_paths = self.load_image_paths()
        random.shuffle(self.img_paths)
        samples_per_class = np.zeros(self.num_classes)
        desired_samples_per_class = int(self.num_samples['train'] / self.num_classes)
        img_paths = list()

        for path in self.img_paths:
            if len(img_paths) >= self.num_samples['train']:
                break

            label = self.path_to_label[path] 
            if samples_per_class[label] < desired_samples_per_class:
                img_paths.append(path)
                samples_per_class[label] += 1
        
        self.num_examples = len(img_paths)
        return img_paths
                            
    def load_labels_to_names(self):
        path = os.path.join(self.root,'category_names_eng.txt')
        label_to_name = list()
        with open(path, 'r') as file:
            for line in file:
                label_to_name.append(line)

        return label_to_name
