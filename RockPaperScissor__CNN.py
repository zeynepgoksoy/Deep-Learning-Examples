#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import keras.layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# In[2]:


base_dir = "/Users/Ap/Desktop/rps-datasets"

train_dir = os.path.join(base_dir,'rps')

paper_dir = os.path.join(train_dir,'paper')
rock_dir = os.path.join(train_dir,'rock')
scissors_dir = os.path.join(train_dir, 'scissors')

paper_imgs = os.listdir(paper_dir)
rock_imgs = os.listdir(rock_dir)
scissors_imgs = os.listdir(scissors_dir)

num_of_labels = len(os.listdir(train_dir))
num_of_paper_imgs = len(os.listdir(paper_dir))
num_of_rock_imgs = len(os.listdir(rock_dir))
num_of_sciss_imgs = len(os.listdir(scissors_dir))

print(num_of_labels)
print(num_of_paper_imgs)
print(num_of_rock_imgs)
print(num_of_sciss_imgs)


# In[3]:


test_dir = os.path.join(base_dir,'rps-test-set')

test_paper_dir = os.path.join(test_dir,'paper')
test_rock_dir = os.path.join(test_dir,'rock')
test_sciss_dir = os.path.join(test_dir,'scissors')

test_paper_imgs = os.listdir(test_paper_dir)
test_rock_imgs = os.listdir(test_rock_dir)
test_sciss_imgs = os.listdir(test_sciss_dir)

test_paper_num = len(os.listdir(test_paper_dir))
test_rock_num = len(os.listdir(test_rock_dir))
test_sciss_num = len(os.listdir(test_sciss_dir))

print(test_paper_num)
print(test_rock_num)
print(test_sciss_num)

test_paper = []
test_rock = []
test_sciss = []

for img in test_paper_imgs:
    imgPath = os.path.join(test_paper_dir,img)
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255
    test_paper.append(image)

nparr_test_paper = np.array(test_paper)

for img in test_rock_imgs:
    imgPath = os.path.join(test_rock_dir,img)
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255
    test_rock.append(image)

nparr_test_rock = np.array(test_rock)

for img in test_sciss_imgs:
    imgPath = os.path.join(test_sciss_dir,img)
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255
    test_sciss.append(image)

nparr_test_sciss = np.array(test_sciss)

X_test = np.append(nparr_test_paper,nparr_test_rock,axis=0)
print(len(X_test))
X_test = np.append(X_test,nparr_test_sciss,axis=0)
print(len(X_test))
print(X_test.shape)
#print(X_train)


# In[4]:


#paper = 0, rock=1, scissors=2
num1 = test_paper_num
num2 = test_paper_num+test_rock_num
num3 = test_paper_num+test_rock_num+test_sciss_num
test_sample_num = num3

label_test = np.ones((test_sample_num,),dtype="float32")
label_test[0:num1] = 0
label_test[num1 : num2] = 1
label_test[num2 : num3] = 2

print(label_test)
Y_test = np_utils.to_categorical(label_test , num_classes = num_of_labels)


# In[5]:


list_paper = []
list_rock =[]
list_scissors = []

number_samples = num_of_paper_imgs + num_of_rock_imgs + num_of_sciss_imgs
labels = np.ones((number_samples,),dtype="float32")


# In[6]:


for img in paper_imgs:
    imgPath = os.path.join(paper_dir,img)  #print(imgPath)
    image=cv2.imread(imgPath)    
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #print(type(image))
    image = image/255 #normalization
    list_paper.append(image)
    #imgvf=cv2.flip(image,flipCode=0) #flip image vertically
    #list_paper.append(imgvf)
    
np_arr_paper = np.array(list_paper)
#print(np_arr)
#print(np_arr.shape)


# In[7]:


for img in rock_imgs:
    imgPath = os.path.join(rock_dir,img)
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255 
    list_rock.append(image)
    #imgvf = cv2.flip(image,flipCode=0)
    #list_rock.append(imgvf)
    
np_arr_rock = np.array(list_rock)


# In[8]:


for img in scissors_imgs:
    imgPath = os.path.join(scissors_dir,img)
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image = image/255
    list_scissors.append(image)
    #imgvf = cv2.flip(image,flipCode=0)
    #list_scissors.append(imgvf)
    
np_arr_scissors = np.array(list_scissors)


# In[9]:


print(len(np_arr_paper))
print(len(np_arr_rock))
print(len(np_arr_scissors))
print(np_arr_paper.shape)
X_train = np.append(np_arr_paper,np_arr_rock,axis=0)
print(len(X_train))
X_train = np.append(X_train,np_arr_scissors,axis=0)
print(len(X_train))
print(X_train.shape)
#print(X_train)


# In[10]:


#paper = 0, rock=1, scissors=2
num1 = num_of_paper_imgs
num2 = num_of_paper_imgs+num_of_rock_imgs
num3 = number_samples

labels[0:num1] = 0
labels[num1 : num2] = 1
labels[num2 : num3] = 2

print(labels)


# In[11]:


X_train = X_train.astype('float32')
Y_train = np_utils.to_categorical(labels , num_classes = num_of_labels)
print(Y_train)


# In[12]:


INPUT_IMG_SHAPE = (X_train.shape)[1:]
print(INPUT_IMG_SHAPE)


# In[13]:


model = Sequential()

# First convolution.
model.add(Conv2D(input_shape=INPUT_IMG_SHAPE, filters=32, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

#second convolution 
model.add(Conv2D( filters=32, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

#third convolution
model.add(Conv2D( filters=32, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

#fourth convolution
model.add(Conv2D(filters=64, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

#fifth convolution
model.add(Conv2D(filters=64, padding='same', kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

#output layer
model.add(Flatten())
model.add(Dense(3,activation='softmax'))


# In[14]:


model.summary()


# In[15]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3,verbose=1)
log_csv =CSVLogger('my_logs2.csv',separator=',',append=True)
model_checkpoint = ModelCheckpoint("My_Trained_model.h5",save_best_only=True,monitor='val_loss',mode='min',verbose=1)

callbacks_list = [early_stop,log_csv,model_checkpoint]


# In[ ]:


#compile model
rmsprop_optimizer = tf.keras.optimizers.RMSprop()

model.compile(
    optimizer=rmsprop_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(X_train.shape)
model.fit(X_train,Y_train,validation_split=0.2,epochs=10,batch_size=32,verbose=2,callbacks=callbacks_list)


# In[ ]:


model.save('rock_paper_scissors_model_after_batch_normalization_and_dropout2.h5')


# In[ ]:


model.save('/Users/Ap/Desktop/saved_models/my_rps_model_after_batchnorm_dropout2')


# In[ ]:


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test, Y_test, batch_size=32)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print("Generate predictions for 3 samples")
#predictions = model.predict(x_test[:3])
#print("predictions shape:", predictions.shape)


# In[ ]:


paper_img_path='/Users/Ap/Desktop/rps-datasets/rps-test-set/paper/testpaper02-03.png'
img_paper = image.load_img(paper_img_path)
plt.imshow(img_paper)
plt.show()

img_paper = np.array(img_paper)
img_paper = img_paper.reshape(1,300,300,3)
print(img_paper.shape)
print(model.predict_classes(img_paper))


# In[ ]:


rock_img_path = '/Users/Ap/Desktop/rps-datasets/rps-test-set/rock/testrock02-11.png'
img_rock =image.load_img(rock_img_path)
plt.imshow(img_rock)
plt.show()

img_rock = np.array(img_rock)
img_rock = img_rock.reshape(1,300,300,3)
print(img_rock.shape)
print(model.predict_classes(img_rock))


# In[ ]:


scissor_img_path = '/Users/Ap/Desktop/makas/scissor1.jpg'
img_scissor = image.load_img(scissor_img_path, target_size=(300, 300))
plt.imshow(img_scissor)
plt.show()

img_scissor = np.array(img_scissor)
img_scissor = img_scissor.reshape(1,300,300,3)
print(model.predict_classes(img_scissor))


# In[ ]:


scissor_img_path = '/Users/Ap/Desktop/makas/scissor2.jpg'
img_scissor = image.load_img(scissor_img_path, target_size=(300, 300))
plt.imshow(img_scissor)
plt.show()

img_scissor = np.array(img_scissor)
img_scissor = img_scissor.reshape(1,300,300,3)
print(model.predict_classes(img_scissor))


# In[ ]:


scissor_img_path = '/Users/Ap/Desktop/rps-datasets/rps-test-set/scissors/testscissors04-30.png'
img_scissor = image.load_img(scissor_img_path)
plt.imshow(img_scissor)
plt.show()

img_scissor = np.array(img_scissor)
img_scissor = img_scissor.reshape(1,300,300,3)
print(model.predict_classes(img_scissor))


# In[ ]:


img_path = '/Users/Ap/Desktop/makas/tas.jpg'
img= image.load_img(img_path, target_size=(300, 300))
plt.imshow(img)
plt.show()

img = np.array(img)
img = img_scissor.reshape(1,300,300,3)
print(model.predict_classes(img))


# In[ ]:




