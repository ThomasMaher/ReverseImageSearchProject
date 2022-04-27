Introudction to Artificial Intelligence Spring 22
# Reverse Visual Search

## Team
- pp2535@nyu.edu - AWS contact
- tm3566@nyu.edu
- ajt9616@nyu.edu

## The Project
### Features
Using a single image as input, the reverse image search will find the n most similar images in a dataset.
### Public Methods
**reverse_image_search(img, output_size):**<br>
&emsp;  Input <br>
&emsp;&emsp; - An image (TODO: currently an index - would like this to be a path or data)<br>
&emsp;&emsp; - The number of desired results. Default is 6.<br>
&emsp;  Function: This will display the n most similar images and return the data for those images.<br>

**TODO: extract_features(data):**<br>
&emsp;  Input<br>
&emsp;&emsp;    - Array of image data<br>
&emsp;  Function: Defines the output of the reverse_image_search function. Takes an array of image data and uses a pretrained ResNet50 model<br> 
&emsp; to extract the features which will be compared to the input of reverse_image_search.
  
### Methodology (?)
The project uses a pretrained CNN model to extract features from a predetermined or user provided (?) set of image data. A k-nearest-neighbours algorithm is used to determine which images in the data are most similar to some input based on the extracted features.<br>
<br>
**CNN Model**<br>
We are using a ResNet50 CNN from Tensorflow Keras pretrained on images from the ImageNet database. The ResNet50 model is a residual convolutional neural network, meaning it uses 'residual blocks' to improve accuracy and prevent degredation (due to vanishing/exploding gradients or model performance) by utilizing skip connections. Skip connections enable the model to learn an identity function, ensuring accuracy across layers (regardless of how far down the gradient goes) and sets up shortcuts for the gradient to pass through (preventing its degredation). This means the model can include a large number of layers (in this case 50). The output of this model is the result of an average pooling on the last convolutional layer. The output of each image is considered to be the extracted features of the image and is saved in an array.<br>
<br>
**KNN**<br>
The K-nearest-beighbours algorithm uses the euclidean distance between datapoints to determine the similarity between different data objects. The k 'nearest' objects are returned. In this case, the euclidean distance is measured between the output from the convolutional layer of an input image and all images from the extracted dataset.
