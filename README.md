# Shifted Gaussian Noise (SGN)
Official implementation of [Robust Classification via Regression-Based Loss Reweighting and Label Correction](https://openreview.net/forum?id=wfgZc3IMqo).

## Abstract
Deep neural networks and large-scale datasets have revolutionized the field of machine learning. However, these large networks are susceptible to overfitting to label noise, resulting in reduced generalization. To address this challenge, two promising approaches have emerged: i) loss reweighting, which reduces the influence of noisy examples on the training loss, and ii) label correction that replaces noisy labels with estimated true labels. These directions have been pursued separately or combined as independent methods, lacking a unified approach. In this work, we present a unified method that seamlessly combines loss reweighting and label correction to enhance robustness against label noise in classification tasks. Specifically, by leveraging ideas from compositional data analysis in statistics, we frame the problem as a regression task, where loss reweighting and label correction can naturally be achieved with a shifted Gaussian label noise model. Our unified approach achieves strong performance compared to recent baselines on several noisy labeled datasets. We believe this work is a promising step towards robust deep learning in the presence of label noise.

## Environment Setup
The code uses and is based on the [Uncertainty Baselines GitHub repository](https://github.com/google/uncertainty-baselines), please follow the installation instructions there.

Our experiments were run with TensorFlow 2.6.0, TensorFlow Probability 0.14.1 and Uncertainty Baselines 0.0.7.

## Running Experiments
For example, the SGN method on CIFAR-100 with 40% symmetric noise, can be run with the following command
```bash
python src/cifar/sgn.py --data_dir=/path/to/data/ \
                        --output_dir=/path/to/output_dir/ \
                        --dataset cifar100 \
                        --noisy_labels \
                        --corruption_type sym \
                        --severity 0.4
```
or on Clothing1M
```bash
python src/clothing1m/sgn.py --data_dir=/path/to/data/ \
                            --output_dir=/path/to/output_dir/ 
```
or on WebVision
```bash
python -u src/webvision/sgn.py --data_dir=/path/to/data/ \
                               --output_dir=/path/to/output_dir/
```