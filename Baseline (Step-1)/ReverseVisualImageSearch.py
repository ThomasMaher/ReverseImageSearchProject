import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import tensorflow
import random
import numpy
from scipy.spatial import distance


# Downloading the dataset
!echo "Downloading lFW dataset"
!curl -L -o lfw.tgz http://vis-www.cs.umass.edu/lfw/lfw.tgz
!tar -xzf lfw.tgz
!rm lfw.tgz

# ResNet50 model
import tensorflow
model = tensorflow.keras.applications.ResNet50(weights='imagenet', include_top=True)

model.summary()

import numpy as np
import matplotlib.pyplot as plt

# Use a path to load and preprocess an image for feature-extraction
def image_numpy(path):
    image1 = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(image1)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return image1, img

image1, img = image_numpy("lfw/Zoe_Ball/Zoe_Ball_0001.jpg")
print("shape of x: ", img.shape)
print("data type: ", img.dtype)
plt.imshow(image1)

labels = model.predict(img)

labels = model(img)  # converts into  a tensorflow object
labels = labels.numpy()   # COnverts that tensorflow object into numpy array

feature_array = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
feature_array.summary()

PathOfImage = 'lfw'
extension = ['.jpg', '.png', '.jpeg']

images_arr = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PathOfImage) for f in filenames if os.path.splitext(f)[1].lower() in extension]

features = []
for i, PathOfImage in enumerate(images_arr):
    if i % 500 == 0:
        print("analyzing image %d: " % (i))
    image1, img = image_numpy(PathOfImage);
    j = feature_array.predict(img)[0]
    features.append(j)

print('finished extracting features for total images: ', len(images_arr))

# Use Euclidean distance to find closest images.
def neighbours(input_image, features, output_size=6):
    '''
    input_image: index in features of image to search
    features: array of 'features' as determined by the CNN for every searchable image
    output_size: number of results returned
    returns: indices (?) of the n closest images based on euclidean distance where n is output_size
    '''
    euclid_distances = [ distance.euclidean(input_image, j) for j in features ]
    nn = sorted(range(len(euclid_distances)), key=lambda k: euclid_distances[k])
    nn = nn[1:output_size + 1] # leaving out image 0 since this should be the same as the input
    return nn

# Format and return images 
def output_nn(images, indexes, result_img_height):
    '''
    data: array of images
    indexes: indices of the images to format
    result_img_height: desired hieght of image
    returns: np array of formatted images
    '''
    result_img = []
    for i in indexes:
        image1 = image.load_img(images[i])
        image1 = image1.resize((int(image1.width * result_img_height / image1.height), result_img_height))
        result_img.append(image1)
    concat_image = np.concatenate([np.asarray(i) for i in result_img], axis=1)
    return concat_image

    import time
# This funciton performs the search. The features array must be an array of the embedded
# features of images as determined by a CNN. Any dataset or model can be used 
# to extract this. The img_path is used to retrieve image data. The image data
# is put through the same CNN that was useed to extract the features array.
# The resulting embedding is used in the nearest neighbors algorithm along with
# the features array to determine the _output_size_ most-similar images.
def reverse_image_search(img_path, features, output_size=6):
  '''
  img_path: path to image to search
  features: array of 'features' as determined by the CNN for every searchable image
  output_size: number of results returned
  returns: array of formated data for n images with the smallest euclidean distance to img 
  based on on features extracted from the pretrained ResNet50 CNN model
  '''
  image1, img = image_numpy(img_path);
  j = feature_array.predict(img)[0]

  nn = neighbours(j, features, output_size)
  Input_images = output_nn([img_path], [0], 300)
  result_array = output_nn(images_arr, nn, 400)

  # display the query image
  plt.figure(figsize = (5,10))
  plt.imshow(Input_images)
  plt.title("query image")

  # display the resulting images
  plt.figure(figsize = (80, 80))
  plt.imshow(result_array)
  plt.title("result images")

  return result_array

  computation_time = []

start = time.time()
reverse_image_search('lfw/Albert_Costa/Albert_Costa_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Angela_Bassett/Angela_Bassett_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Arminio_Fraga/Arminio_Fraga_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Billy_Crystal/Billy_Crystal_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Bob_Graham/Bob_Graham_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Boris_Becker/Boris_Becker_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Bulent_Ecevit/Bulent_Ecevit_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Calista_Flockhart/Calista_Flockhart_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Cameron_Diaz/Cameron_Diaz_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

start = time.time()
reverse_image_search('lfw/Carmen_Electra/Carmen_Electra_0001.jpg', features, 20)
t = time.time()
computation_time.append(t - start)

print(f'Average computation time: {sum(computation_time)/len(computation_time)}')
