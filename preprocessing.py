from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm
import keras
import numpy as np
import pandas as pd
import cv2
import imutils 
import os
import random
import math


# In[ ]:


from google.colab import drive

drive.mount('/content/gdrive')


# In[ ]:


BASE_PATH = '/content/gdrive/MyDrive/Colab Notebooks/Brain Tumor Classification/'
PATH = BASE_PATH + 'original_dataset/'

AUGMENTED_TRAIN_PATH_YES = BASE_PATH + 'augmented_dataset/train/yes/'
AUGMENTED_TRAIN_PATH_NO = BASE_PATH + 'augmented_dataset/train/no/'

AUGMENTED_VALIDATION_PATH_YES = BASE_PATH + 'augmented_dataset/validation/yes/'
AUGMENTED_VALIDATION_PATH_NO = BASE_PATH + 'augmented_dataset/validation/no/'

AUGMENTED_TEST_PATH_YES = BASE_PATH + 'augmented_dataset/test/yes/'
AUGMENTED_TEST_PATH_NO = BASE_PATH + 'augmented_dataset/test/no/'

TARGET_IMAGE_SIZE = (224, 224)


# In[ ]:


dataset = []


# In[ ]:


for feature_class in tqdm(os.listdir(PATH)):
  if not feature_class.startswith("."):
      feature_class_path = PATH + feature_class
      for (i, file_name) in enumerate(os.listdir(feature_class_path)):
          file_path = feature_class_path + '/' + file_name
          image = cv2.imread(file_path)
          image_and_class = (image, feature_class)
          dataset.append(image_and_class)


# In[ ]:


random.shuffle(dataset)


# In[ ]:


TRAINING_SAMPLES_SIZE = 0.7
VALIDATION_SAMPLES_SIZE = 0.15
TESTING_SAMPLES_SIZE = 0.15


# In[ ]:


if (TRAINING_SAMPLES_SIZE + VALIDATION_SAMPLES_SIZE + TESTING_SAMPLES_SIZE) > 100:
    raise ValueError


# In[ ]:


total_dataset_size = len(dataset)

training_start_index = 0
training_end_index = training_start_index + math.floor(total_dataset_size * TRAINING_SAMPLES_SIZE) 

validation_start_index = training_end_index
validation_end_index = validation_start_index + math.floor(total_dataset_size * VALIDATION_SAMPLES_SIZE) 

testing_start_index = validation_end_index
testing_end_index = testing_start_index + math.floor(total_dataset_size * TESTING_SAMPLES_SIZE)


# In[ ]:


training_dataset = dataset[training_start_index:training_end_index]
validation_dataset = dataset[validation_start_index:validation_end_index]
testing_dataset = dataset[testing_start_index:testing_end_index]


# In[ ]:


def count_class_labels(dataset, dataset_type):
    yes_count = 0
    no_count = 0
    for data in dataset:
        label = data[1]
        if label == 'yes':
            yes_count = yes_count + 1
        else:
            no_count = no_count + 1
    print("Number of YES labels in the {0} dataset are {1}".format(dataset_type, yes_count))
    print("Number of NO labels in the {0} dataset are {1}".format(dataset_type, no_count))
    print("---")


# In[ ]:


count_class_labels(training_dataset, "training")
count_class_labels(validation_dataset, "validation")
count_class_labels(testing_dataset, "testing")


# In[ ]:


def reshape_image_array(image):
    '''
    Reshapes the image numpy array to make it four dimension since ImageDataGenerator requires a four dimensioned array
    '''
    return image.reshape((1,) + image.shape)


# In[ ]:


def resize_image(image, target_image_size=TARGET_IMAGE_SIZE):
    resized = cv2.resize(image, dsize=TARGET_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized


# In[ ]:


def split_yes_no_dataset(dataset):
    yes = []
    no = []
    for data in tqdm(dataset):
        label = data[1]
        image = data[0]
        resized = resize_image(image)
        reshaped = reshape_image_array(resized)
        if label == "yes":
            yes.append(reshaped)
        elif label == "no":
            no.append(reshaped)
    return yes, no


# In[ ]:


training_yes_dataset, training_no_dataset = split_yes_no_dataset(training_dataset)


# In[ ]:


validation_yes_dataset, validation_no_dataset = split_yes_no_dataset(validation_dataset)


# In[ ]:


testing_yes_dataset, testing_no_dataset = split_yes_no_dataset(testing_dataset)


# In[ ]:


def augment_images(dataset, output_path):
    if not os.path.exists(output_path):
          os.makedirs(output_path)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
    )
    image_count = 0
    for image in tqdm(dataset):
        image_count = image_count + 1
        generator = datagen.flow(
            image,
            save_to_dir=output_path
            )
        iteration = 0
        for batch in generator:
            iteration = iteration + 1
            if iteration == 5:
              break


# In[ ]:


augment_images(dataset=training_yes_dataset, output_path=AUGMENTED_TRAIN_PATH_YES)


# In[ ]:


augment_images(dataset=training_no_dataset, output_path=AUGMENTED_TRAIN_PATH_NO)


# In[ ]:


augment_images(dataset=validation_yes_dataset, output_path=AUGMENTED_VALIDATION_PATH_YES)


# In[ ]:


augment_images(dataset=validation_no_dataset, output_path=AUGMENTED_VALIDATION_PATH_NO)


# In[ ]:


augment_images(dataset=testing_yes_dataset, output_path=AUGMENTED_TEST_PATH_YES)


# In[ ]:


augment_images(dataset=testing_no_dataset, output_path=AUGMENTED_TEST_PATH_NO)

