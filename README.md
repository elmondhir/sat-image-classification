# Exploring Satellite Image Classification with Deep Learning

Recent advances in deep learning have enabled significant progress in computer vision tasks like image classification. One particularly interesting application is classifying satellite imagery, which has many real-world use cases from urban planning to environmental monitoring. In this post, I'll walk through my experience training and evaluating several popular convolutional neural network (CNN) architectures on the RSI-CB256 satellite image dataset.

## The Dataset
The RSI-CB256 dataset contains 36,288 satellite images covering four different classes:

- Desert
- Green Area
- Water
- Cloudy

The images were sourced from remote sensing imagery as well as Google Maps snapshots, with each class containing a mix from both sources. At 256x256 pixels, the images are fairly high resolution, allowing the models to pick up on detailed textures and patterns.

Here's an example image from each class:

Ex1 
EX2

## Data Preparation
The RSI-CB256 dataset comes pre-organized with the images separated into four folders corresponding to the four classes: Desert, Green Area, water, and Cloudy. This made it easy to use Keras `ImageDataGenerator` to efficiently load images during training and evaluation.

I split the full dataset into 80% train, 10% validation, and 10% test sets using scikit-learn's train_test_split function. Then I created data generators for each split that could load images in batches, apply data augmentation, and preprocess the images on the fly:
