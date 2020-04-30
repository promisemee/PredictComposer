#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import keras

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from datetime import datetime 
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D, GRU, Reshape, BatchNormalization

#Fetch csv files
test_dir = 'test_image'
train_dir = 'train_image'

train_csv = "csv/train_data.csv"
test_csv = "csv/test_data.csv"

train = pd.DataFrame(data=pd.read_csv(train_csv,dtype = str, error_bad_lines=False))
test = pd.DataFrame(data=pd.read_csv(test_csv,dtype = str, error_bad_lines=False))

#Drop composerse out of list
List = ['Ludwig van Beethoven', 'Wolfgang Amadeus Mozart', 'Johann Sebastian Bach', 'Franz Schubert', 'Frédéric Chopin']
train = train[train['composer'].isin(List)]
test = test[test['composer'].isin(List)]

#Create datagenerator
traingen=ImageDataGenerator(rescale=1./255, validation_split = 0.1)
testgen=ImageDataGenerator(rescale=1./255)

train_generator=traingen.flow_from_dataframe(
      dataframe=train,
      directory = "train_image/",
      x_col="image_file",
      y_col="composer",
      subset="training",
      batch_size=32,
      seed=42,
      shuffle=True,
      class_mode="categorical",
      target_size=(32,96))

validation_generator = traingen.flow_from_dataframe(
    dataframe=train,
    directory = "train_image/",
    x_col="image_file",
    y_col="composer",
    subset="validation",
    target_size=(32,96),
    batch_size=32,
    class_mode='categorical')

test_generator=testgen.flow_from_dataframe(
  dataframe=test,
  directory = "test_image/",
  x_col="audio_file",
  y_col="composer",
  batch_size=32,
  seed=42,
  shuffle=True,
  class_mode="categorical",
  target_size=(32,96))

##SVM

# One hot encoder for SVM
from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder(handle_unknown='ignore') 
enc_df = pd.DataFrame(onehotencoder.fit_transform(train[['composer']]).toarray())
train = train.join(enc_df)
onehotencoder = OneHotEncoder(handle_unknown='ignore') 
enc_df = pd.DataFrame(onehotencoder.fit_transform(test[['composer']]).toarray())
test = test.join(enc_df)

from pathlib import Path

#train_image
folder = Path("train_image/")
dirs = folder.glob("*")
labels_dict = {'Johann Sebastian Bach': 3 , 'Frédéric Chopin': 2, 'Wolfgang Amadeus Mozart': 4, 
               'Ludwig van Beethoven': 0, 'Franz Schubert': 1}

image_data = []
labels = [] 

for index, row in train.iterrows():
    img_path = Path(path+"train_image2/"+row["image_file"])
    img = image.load_img(img_path, target_size = (32,96))
    img_array = image.img_to_array(img)
    image_data.append(img_array)
    label = row[[2,3,4,5,6]].to_numpy()
    labels.append(label)

## Convert data into numpy array

image_data = np.array(image_data, dtype='float32')/255.0
labels = np.array(labels)

#test_image
folder = Path(path+"test_image/")
dirs = folder.glob("*")

test_data = []
test_labels = [] 

for index, row in test.iterrows():
    img_path = Path(path+"test_image/"+row["audio_file"])
    img = image.load_img(img_path, target_size = (32,96))
    img_array = image.img_to_array(img)
    test_data.append(img_array)
    label = row[[2,3,4,5,6]].to_numpy()
    test_labels.append(label)

## Convert data into numpy array

test_data = np.array(test_data, dtype='float32')/255.0
test_labels = np.array(test_labels)

image_data = image_data.reshape(11633, 32*96*3)
test_data = test_data.reshape(150, 32*96*3)
print(image_data.shape, test_data.shape)
print(labels.shape)

#Train SVM
from sklearn.svm import LinearSVC
clf = linearSVC()
labels = np.argmax(labels, axis = 1)
clf.fit(image_data, labels)
y_pred = clf.predict(image_data)
clf.score(image_data, labels)

#Predict SVM
clf.score(test_data, test_labels)

# Plot confusion matrix

y_pred = clf.predict(test_data)
cm = confusion_matrix(y_pred, test_labels)
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
       plt.text(j, i, z, ha='center', va='center')
plt.show()

# # CNN

# In[17]:


num_labels = 5
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = test_generator.n//test_generator.batch_size


# In[15]:


def buildCNN(layers, num_labels, weight=None):
    model = Sequential()
    
    channel_axis = 3
    
    model.add(Conv2D(layers[0], (3, 3), strides=(1, 1), input_shape = (32,96,3), padding = "same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(layers[1], (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(layers[2], (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layers[3], (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
        
    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))

    if weight is None:
        return model

    else:
        model.load_weights(weight)
        return model


# In[21]:


#CNN Model
layers = [8,8,16,16]
model = buildCNN(layers, num_labels)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[23]:


#Train CNN
start = datetime.now()

num_epochs = 150
checkpointer1 = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn2301_test.hdf5', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator, 
                              validation_steps = STEP_SIZE_VALID, callbacks = [checkpointer1, es], epochs=num_epochs)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save('saved_models/cnn2301.h5')


# In[30]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[5]:


from tensorflow.keras.models import load_model
model = load_model('saved_models/cnn2301.h5')


# In[7]:


y_pred = model.predict_generator(validation_generator)


# In[12]:


np.equal(validation_generator.classes, np.argmax(y_pred, axis = 1)).mean()


# In[14]:


#Confusion Matrix

Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)

# Plot confusion matrix
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
       plt.text(j, i, z, ha='center', va='center')
plt.show()


# In[18]:


#CNN Model
layers = [32, 64, 128, 128]
modelcnn2 = buildCNN(layers, num_labels)
modelcnn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelcnn2.summary()


# In[19]:


#Train CNN
start = datetime.now()

num_epochs = 150

cp1 = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn2302_test.hdf5', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
history2 = modelcnn2.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator, 
                              validation_steps = STEP_SIZE_VALID, callbacks = [cp1, es], epochs=num_epochs)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save('saved_models/cnn2302.h5')


# In[34]:


plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[20]:


plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[21]:


plt.plot(history.history['acc'])
plt.plot(history2.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train1','train2','test1', 'test2'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[28]:


print(history2.history['acc'][-1])
print(history2.history['val_acc'][-1])
print(history2.history['loss'][-1])
print(history2.history['val_loss'][-1])


# In[47]:


print(modelcnn2.evaluate_generator(train_generator))
print(modelcnn2.evaluate_generator(validation_generator))
print(modelcnn2.evaluate_generator(test_generator))
print()
modeltest = load_model('saved_models/weights.best.basic_cnn2302_test.hdf5')
print(modeltest.evaluate_generator(train_generator))
print(modeltest.evaluate_generator(validation_generator))
print(modeltest.evaluate_generator(test_generator))


# In[41]:


def confusion_matrix(model, generator):

    Y_pred = model.predict_generator(generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(generator.classes, y_pred)
    
    # Plot confusion matrix
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
           plt.text(j, i, z, ha='center', va='center')
    plt.show()


# In[48]:


confusion_matrix(modelcnn2, test_generator)


# In[49]:


#Confusion Matrix

Y_pred = modelcnn2.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)

# Plot confusion matrix
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
       plt.text(j, i, z, ha='center', va='center')
plt.show()


# In[ ]:





# In[ ]:


#Confusion Matrix

Y_pred = model2.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)

# Plot confusion matrix
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
       plt.text(j, i, z, ha='center', va='center')
plt.show()


# In[ ]:


import numpy as np
from keras.preprocessing import image

img_width, img_height = 32, 32
img = image.load_img('/home/dain/IndividualProject/train_image/train_dataset0.png', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

p = model.predict(img)
y_classes = p.argmax(axis=0)
y_classes


# In[6]:


print(history.history.keys())


# In[ ]:


plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # CRNN

# In[22]:


def buildCRNN(num_labels, weight=None):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape = (32,96,3), padding = "same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D((3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.1))
    
    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Reshape((16,128)))
    
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(rate=0.3))
    
    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))
    
    if weight is None:
        return model

    else:
        model.load_weights(weight)
        return model


# In[50]:


#Build CRNN
model2 = buildCRNN(num_labels)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()


# In[ ]:


#Train CRNN
start = datetime.now()
num_epochs = 150

checkpointer2 = ModelCheckpoint(filepath='saved_models/weights.best.basic_crnn2301_test.hdf5', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
history3 = model2.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator, 
                                validation_steps=STEP_SIZE_VALID, callbacks = [checkpointer2, es], epochs=num_epochs)
duration = datetime.now() - start
print("Training completed in time: ", duration)
model2.save('saved_models/crnn2301.h5')


# In[ ]:


#Confusion Matrix
Y_pred = model2.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm2 = confusion_matrix(test_generator.classes, y_pred)

# Plot confusion matrix
plt.imshow(cm2,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm2):
       plt.text(j, i, z, ha='center', va='center')
plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()