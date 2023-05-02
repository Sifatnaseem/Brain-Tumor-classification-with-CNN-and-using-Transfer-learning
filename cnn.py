from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import OneHotEncoder 
from tqdm import tqdm
from keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random


# In[18]:


from google.colab import drive

drive.mount('/content/gdrive')


# In[19]:


BASE_PATH = '/content/gdrive/MyDrive/Colab Notebooks/Brain Tumor Classification/'

AUGMENTED_TRAIN_PATH = BASE_PATH + 'augmented_dataset/train/'

AUGMENTED_VALIDATION_PATH = BASE_PATH + 'augmented_dataset/validation/'

AUGMENTED_TEST_PATH = BASE_PATH + 'augmented_dataset/test/'

TARGET_IMAGE_SIZE = (224, 224)


# In[20]:


def read_dataset(path):
    dataset = []
    for feature_class in tqdm(os.listdir(path)):
      if not feature_class.startswith("."):
          feature_class_path = path + feature_class
          for (i, file_name) in enumerate(os.listdir(feature_class_path)):
              file_path = feature_class_path + '/' + file_name
              image = cv2.imread(file_path)
              image_and_class = (image, feature_class)
              dataset.append(image_and_class)
    return dataset


# In[21]:


training_dataset = read_dataset(AUGMENTED_TRAIN_PATH)
random.shuffle(training_dataset)


# In[22]:


validation_dataset = read_dataset(AUGMENTED_VALIDATION_PATH)
random.shuffle(validation_dataset)


# In[23]:


testing_dataset = read_dataset(AUGMENTED_TEST_PATH)
random.shuffle(testing_dataset)


# In[24]:


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


# In[25]:


count_class_labels(training_dataset, "training")
count_class_labels(validation_dataset, "validation")
count_class_labels(testing_dataset, "testing")


# In[26]:


encoder = OneHotEncoder()
encoder.fit([[0], [1]]) 


# In[27]:


def split_into_X_and_y(dataset):
    X = []
    y = []
    for data in dataset:
        X.append(data[0])
        label = data[1]
        if label == 'yes':
            y.append(encoder.transform([[1]]).toarray())
        elif label == 'no':
            y.append(encoder.transform([[0]]).toarray())

    return np.array(X), np.array(y).reshape(len(y), 2)


# In[28]:


X_train, y_train = split_into_X_and_y(training_dataset)
X_validation, y_validation = split_into_X_and_y(validation_dataset)
X_test, y_test = split_into_X_and_y(testing_dataset)


# In[39]:


KERNEL_SIZE = (2, 2)
STRIDES = (2, 2)
ACTIVATION_FUNCTION_RELU = 'relu'
PADDING_SAME = 'same'
DROPOUT = 0.25

model = Sequential()

# Block 1
model.add(Conv2D(32, kernel_size = KERNEL_SIZE, padding = PADDING_SAME, input_shape = (224, 224, 3)))
model.add(Conv2D(32, kernel_size = KERNEL_SIZE,  activation = ACTIVATION_FUNCTION_RELU, padding = PADDING_SAME))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(DROPOUT))

# Block 2
model.add(Conv2D(64, kernel_size = KERNEL_SIZE, activation = ACTIVATION_FUNCTION_RELU, padding = PADDING_SAME))
model.add(Conv2D(64, kernel_size = KERNEL_SIZE, activation = ACTIVATION_FUNCTION_RELU, padding = PADDING_SAME))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = KERNEL_SIZE, strides = STRIDES))
model.add(Dropout(DROPOUT))

# Block 3
model.add(Conv2D(128, kernel_size = KERNEL_SIZE, activation = ACTIVATION_FUNCTION_RELU, padding = PADDING_SAME))
model.add(Conv2D(128, kernel_size = KERNEL_SIZE, activation = ACTIVATION_FUNCTION_RELU, padding = PADDING_SAME))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = KERNEL_SIZE, strides = STRIDES))
model.add(Dropout(DROPOUT))

model.add(Flatten())

model.add(Dense(512, activation = ACTIVATION_FUNCTION_RELU))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])


# In[40]:


model.summary()


# In[41]:


result = model.fit(X_train, y_train, epochs = 5, batch_size = 50, verbose = 1, validation_data = (X_validation, y_validation))


# In[42]:


loss, accuracy = model.evaluate(X_test, y_test)


# In[43]:


accuracy

