import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from natsort import natsorted
import glob


# number of class
classTotal = 5

classNumb = 5

testImagePath = 'D:/SN/klafikasiSawah/dataset/test/' + str(classNumb) + '/'

# testImagePath = 'D:/SN/Data to Zasya/random/'

train = ImageDataGenerator(rescale = 1/255)

trainData = train.flow_from_directory('D:/SN/klafikasiSawah/dataset/train/',
                                      target_size = (200, 200),
                                      batch_size = 3, 
                                      # classes = 
                                      class_mode = 'categorical')

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (200, 200, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),                                
                                    #
                                    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),                                
                                    #                                     
                                    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),                                    
                                    #                                    
                                    tf.keras.layers.Flatten(),                                    
                                    # 
                                    tf.keras.layers.Dense(512, activation = 'relu'),                                    
                                    #                                    
                                    tf.keras.layers.Dense(classTotal, activation = 'softmax')
                                    
    ])


model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])

model.load_weights('D:/SN/klafikasiSawah/result/klasifikasiSawahv2epoch500.h5')
# print(model.get_weights())

# loss, acc = model.evaluate(trainData)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

totalKelas1 = 0
totalKelas2 = 0
totalKelas3 = 0
totalKelas4 = 0
totalKelas5 = 0
for filename in natsorted(glob.glob(testImagePath + '*.png'), key = len): # path to your images folder
    print(filename)
    imageForPredict = image.load_img(filename, target_size = (200, 200))
    # plt.imshow(imageForPredict)
    # plt.show()
    # print(trainData.class_indices)
    imageArray = image.img_to_array(imageForPredict)
    imageArray = np.expand_dims(imageArray, axis = 0)
    # print(imageArray)
    imageRes = np.vstack([imageArray])
    # print(imageRes)
    # val = np.argmax([imageRes])
    val = model.predict(imageRes)
    # print(val)
    valPredict = np.argmax([val])
    print("Kelas: " + str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]))

    if (str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]) == '1'):
        totalKelas1 = totalKelas1 + 1
    elif (str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]) == '2'):
        totalKelas2 = totalKelas2 + 1
    elif (str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]) == '3'):
        totalKelas3 = totalKelas3 + 1
    elif (str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]) == '4'):
        totalKelas4 = totalKelas4 + 1
    elif (str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]) == '5'):
        totalKelas5 = totalKelas5 + 1

print('total kelas 1 = ' + str(totalKelas1))
print('total kelas 2 = ' + str(totalKelas2))
print('total kelas 3 = ' + str(totalKelas3))
print('total kelas 4 = ' + str(totalKelas4))
print('total kelas 5 = ' + str(totalKelas5))