""" protobuf=3.19.0, """

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
# from keras.utils.vis_utils import plot_model

#tensorboad
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import matplotlib.pyplot as plt
import cv2
import numpy as np
import datetime, os

# number of class
classNumb = 5


#normalization
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

#labelingImage
trainData = train.flow_from_directory('D:/SN/klafikasiSawah/dataset/train/',
                                      target_size = (200, 200),
                                      batch_size = 3, 
                                      # classes = 
                                      class_mode = 'categorical')


validationData = validation.flow_from_directory('D:/SN/klafikasiSawah/dataset/val/',
                                      target_size = (200, 200),
                                      batch_size = 3, 
                                      class_mode = 'categorical')
# print(trainData.class_indices)
# print(trainData.classes)

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
                                    tf.keras.layers.Dense(classNumb, activation = 'softmax') 
                                    
    ])

print(model.summary())

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])

# inisialisasi tensorboard
# tbColab = TensorBoardColab()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Jalankan tensorboard -->  python -m tensorboard.main --logdir=logs/


model_fit = model.fit(trainData,
                      batch_size=2,
                      steps_per_epoch = 4,
                      epochs = 500,
                      validation_data = validationData,
                      callbacks=[tensorboard_callback])
                    #   EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
model.save_weights('D:/SN/klafikasiSawah/result/klasifikasiSawahv2epoch500.h5')