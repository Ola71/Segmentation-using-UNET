#----------------------------------------------------------------------
#                     import Libraries
#----------------------------------------------------------------------
import unet_archi
from unet_archi import build_unet
import cv2
import time
import datetime
from datetime import datetime
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #To eleminate masseges about RAM & GPU
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import keras
from tensorflow.python.keras.utils import conv_utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#----------------------------------------------------------------------
#                       Data Directories
#----------------------------------------------------------------------
Train_images_dir="C:/Kaggle_dataset_Splited/train_images"
Train_mask_dir ="C:/Kaggle_dataset_Splited/train_masks"
Val_images_dir = "C:/Kaggle_dataset_Splited/val_images"
Val_mask_dir ="C:/Kaggle_dataset_Splited/val_masks"
Test_image_dir="C:/Kaggle_dataset_Splited/test_images"
Test_mask_dir="C:/Kaggle_dataset_Splited/test_masks"
images_in_train_images="C:/Kaggle_dataset_Splited/train_images/images"
My_model_save_link='C:/Results_UNet/UNet_results/Lung_segmentation_UNet_load_from_disk.hdf5'
Results_Link_Net="C:/Results_Link_Net"
#----------------------------------------------------------------------
#-------------- Global Parameters -------------------------------------
Size=256
seed=24   
batch_size=4 
#---- Augument Parameters images -----

img_data_gen_args = dict(rescale = 1/255.0,  
                      rotation_range=25,
                      width_shift_range=3,
                      height_shift_range=3,
                      shear_range=0.1,#image will be distorted along an axis
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=False,
                      fill_mode='reflect')# nearest  or  reflect or wrap or constant
                      

#---- Augument Parameters Masks -----
mask_data_gen_args = dict(#rescale = 1/255.0,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                      rotation_range=25,
                      width_shift_range=3,
                      height_shift_range=3,
                      shear_range=0.1,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=False,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) 

#----- Apply the Augmentation for images and masks

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory(Train_images_dir, 
                                                           seed=seed, 
                                                           batch_size=batch_size,
                                                            target_size=(Size, Size),
                                                           # save_to_dir='C:/Users/olamo/OneDrive/Desktop/Theisis/Thesis_Progs/Results/res4/test',
                                                           class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                            #thinking class mode is binary.
print(image_generator)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

mask_generator = mask_data_generator.flow_from_directory(Train_mask_dir, 
                                                         seed=seed, 
                                                         batch_size=batch_size,
                                                         target_size=(Size, Size),
                                                         #save_to_dir='C:/Users/olamo/OneDrive/Desktop/Theisis/Thesis_Progs/Results/res4/testmask',
                                                         color_mode = 'grayscale',   #Read masks in grayscale
                                                         class_mode=None)
print(mask_generator)

#----- also use generator to validation but without augumentation


valid_img_generator = image_data_generator.flow_from_directory(Val_images_dir, 
                                                               seed=seed, 
                                                               batch_size=batch_size, 
                                                               target_size=(Size, Size),
                                                               class_mode=None) #Default batch size 32, if not specified here
valid_mask_generator = mask_data_generator.flow_from_directory(Val_mask_dir, 
                                                               seed=seed, 
                                                               batch_size=batch_size, 
                                                                target_size=(Size, Size),
                                                               color_mode = 'grayscale',   #Read masks in grayscale
                                                               class_mode=None)  #Default batch size 32, if not specified here




train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)


print(valid_img_generator)
print(valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()

IMG_HEIGHT = x.shape[1] #256
IMG_WIDTH  = x.shape[2] #256
IMG_CHANNELS = x.shape[3] #3


# input_shape = (256, 256,3)

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model =  build_unet(input_shape)

from focal_loss import BinaryFocalLoss

metrics=['accuracy']  


model.compile(optimizer=Adam(lr = 1e-4), loss=BinaryFocalLoss(gamma=2), 
              metrics=metrics)


model.summary()

num_train_imgs = len(os.listdir(images_in_train_images))

print('number of train images',num_train_imgs)
steps_per_epoch = num_train_imgs //batch_size

print('number of steps_per_epoch',steps_per_epoch)

my_callbacks = [                
        #tf.keras.callbacks.EarlyStopping(patience=25),
        tf.keras.callbacks.ModelCheckpoint(filepath=Results_Unet +'model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                ]
time_start = datetime.now() 

history = model.fit_generator(train_generator, validation_data=val_generator, 
                    steps_per_epoch=steps_per_epoch, callbacks= my_callbacks,
                    validation_steps=steps_per_epoch, epochs=100)

print('Time of Training', datetime.now() - time_start)

np.save(Results_Unet+'_U_NET_history.npy',history.history)
# history=np.load(Results_Unet+'_U_NET_history.npy',allow_pickle='TRUE').item()

#----------------------------------------------------------------------
#                         MODEL  SAVE
#----------------------------------------------------------------------

model.save(My_model_save_link)


#----------------------------------------------------------------------
#plot the training and validation accuracy and loss at each epoch
#----------------------------------------------------------------------

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# plt.style.use('seaborn-whitegrid')
plt.plot(epochs, loss,  label='Training loss',color='#059DC0')
plt.plot(epochs, val_loss,  label='Validation loss',color='#F652A0')
plt.title('Training and validation loss - Unet',fontsize=20,color='#44444C')
plt.xlabel('Epochs',fontsize=14,color='#44444C')
plt.ylabel('Loss',fontsize=14,color='#44444C')
plt.grid(True)
plt.legend(prop={'size': 16},frameon=True)
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']



# plt.style.use('seaborn-whitegrid')#'Accuracy'
plt.plot(epochs, acc,  label='Accuracy',color='#FFA384')#Fushi #F50CA0
plt.plot(epochs, val_acc,  label='Validation Accuracy',color='#81B622')
plt.title('Training and validation Accuracy- Unet',fontsize=20,color='#44444C')
plt.xlabel('Epochs',fontsize=14,color='#44444C')
plt.ylabel('Accuracy',fontsize=14,color='#44444C')
plt.grid(True)
plt.legend(prop={'size': 16},frameon=True)
plt.show()





model = tf.keras.models.load_model(My_model_save_link, compile=False)




test_data_gen_args = dict(rescale = 1/255.0 )


test_data_generator = ImageDataGenerator(**test_data_gen_args)
test_img_generator = test_data_generator.flow_from_directory(Test_image_dir,target_size = (SIZE, SIZE),  
                                                              seed=seed, 
                                                              batch_size=71, 
                                                              class_mode=None) #Default batch size 32, if not specified here


test_mask_generator = test_data_generator.flow_from_directory(Test_mask_dir,target_size = (SIZE, SIZE),  
                                                              seed=seed, 
                                                              batch_size=71, 
                                                              color_mode = 'grayscale',   #Read masks in grayscale
                                                              class_mode=None,
                                                              )


#----------------------------------------------------------------------
# Testing on a few test images
#----------------------------------------------------------------------
a = test_img_generator.next()
b = test_mask_generator.next()


# import seaborn
# seaborn.set_style(style=None)
for i in range(0,5):# Plot 6 images at a time
    image = a[i]
    mask = b[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()


import random
test_img_number = random.randint(0, a.shape[0]-1)
test_img = a[test_img_number]
ground_truth=b[test_img_number]
test_img_input=np.expand_dims(test_img, 0)



prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
# any value under 0.5 is background else it is Lungs


# plt.style.use('seaborn-white')
plt.figure(figsize=(16, 8))
plt.subplot(241)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(242)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(243)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')


gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img_norm=cv2.bitwise_and(gray,gray,mask=prediction)
test_img_norm=cv2.cvtColor(test_img_norm,cv2.COLOR_GRAY2RGB  )
plt.subplot(244)
plt.title('Segmented Lung')
plt.imshow(test_img_norm)

plt.show()
#========================================================

#----------------------------------------------------------------------
#                    Model EVALUATION  for All testing images
#----------------------------------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.metrics import MeanIoU 


time_start = datetime.now()

n_classes = 2
SCORE = []
IoU_values = []
for img in range(0, a.shape[0]):
    temp_img = a[img]
    ground_truth=b[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)
    
    img_path=os.path.basename(str(test_img_generator.filepaths))
    
    """ Calculating metrics values """
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    acc_value = accuracy_score(ground_truth, prediction)
    f1_value = f1_score(ground_truth, prediction, labels=[0, 1], average="binary")
    jac_value = jaccard_score(ground_truth, prediction, labels=[0, 1], average="binary")
    recall_value = recall_score(ground_truth, prediction, labels=[0, 1], average="binary")
    precision_value = precision_score(ground_truth, prediction, labels=[0, 1], average="binary")
    SCORE.append([img_path, acc_value, f1_value, jac_value, recall_value, precision_value,IoU])


    
    


print('Number of testesd images = ',a.shape[0])
print('Time of Testing : ', datetime.now() - time_start)

# N O T E that batch_size is 72 for test for calculating the IOU   




df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
 

""" Metrics values """
score = [s[1:]for s in SCORE]
score = np.mean(score, axis=0)
print(f"Accuracy: {score[0]:0.5f}")
print(f"F1: {score[1]:0.5f}")
print(f"Jaccard: {score[2]:0.5f}")
print(f"Recall: {score[3]:0.5f}")
print(f"Precision: {score[4]:0.5f}")
print(f"Mean IoU : {mean_IoU[0]:0.5f}" ) 

""" Saving all the results """
df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision","Mean_IOU"])
df.to_csv("UNet_score.csv")

#-------------------------------      E N D    --------------------------





