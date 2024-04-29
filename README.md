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

## Model Architectures
I decided to test three well-known CNN model families that have shown strong performance on image classification tasks:

**EfficientNet:** Developed by Google Brain, this model scales up existing architectures like MobileNets and achieves better accuracy and efficiency through compound scaling of dimensions like depth, width and resolution. I used the EfficientNetB0 variant.

**ResNet:** The residual network architecture from Microsoft introduced skip connections to allow easier training of very deep networks (up to 152 layers in ResNet-152). I used the ResNet50 variant which is 50 layers deep.

VGG: The VGG network from Oxford prioritizes depth over width, with very small 3x3 convolutional filters stacked to achieve a large effective receptive field. I used the VGG16 variant with 16 weight layers.
