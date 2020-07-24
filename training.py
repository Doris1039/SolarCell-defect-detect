import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras import datasets, layers, models,regularizers
import matplotlib.pyplot as plt

import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

def getInfo(train_dir,test_dir):
    train_Break_dir = os.path.join(train_dir, 'Break')  # directory with our training cat pictures
    train_Cellfaults_dir = os.path.join(train_dir, 'Cell-faults')  # directory with our training dog 
    train_Fingerinteruptions_dir = os.path.join(train_dir, 'Finger-interuptions')  # directory with our training cat pictures
    train_Functional_dir = os.path.join(train_dir, 'Functional')  # directory with our training dog pictures
    train_Materialdefect_dir = os.path.join(train_dir, 'Material-defect')  # directory with our training cat pictures
    train_Microcrack_dir = os.path.join(train_dir, 'Microcrack')  # directory with our training dog pictures
    train_Darkspot_dir = os.path.join(train_dir, 'Dark-spot')  # directory with our validation dog pictures

    validation_Break_dir = os.path.join(validation_dir, 'Break')  # directory with our validation cat pictures
    validation_Cellfaults_dir = os.path.join(validation_dir, 'Cell-faults')  # directory with our validation dog 
    validation_Fingerinteruptions_dir = os.path.join(validation_dir, 'Finger-interuptions')  # directory with our validation cat pictures
    validation_Functional_dir = os.path.join(validation_dir, 'Functional')  # directory with our validation dog pictures
    validation_Materialdefect_dir = os.path.join(validation_dir, 'Material-defect')  # directory with our validation cat pictures
    validation_Microcrack_dir = os.path.join(validation_dir, 'Microcrack')  # directory with our validation dog pictures
    validation_Darkspot_dir = os.path.join(validation_dir, 'Dark-spot')  # directory with our validation dog pictures

    num_Fingerinteruptions_tr = len(os.listdir(train_Fingerinteruptions_dir))
    num_Materialdefect_tr = len(os.listdir(train_Materialdefect_dir))
    num_Break_tr = len(os.listdir(train_Break_dir))
    num_Cellfaults_tr = len(os.listdir(train_Cellfaults_dir))
    num_Functional_tr = len(os.listdir(train_Functional_dir))
    num_Microcrack_tr = len(os.listdir(train_Microcrack_dir))
    num_Darkspot_tr = len(os.listdir(train_Darkspot_dir))

    num_Fingerinteruptions_val = len(os.listdir(validation_Fingerinteruptions_dir))
    num_Materialdefect_val = len(os.listdir(validation_Materialdefect_dir))
    num_Break_val = len(os.listdir(validation_Break_dir))
    num_Cellfaults_val = len(os.listdir(validation_Cellfaults_dir))
    num_Functional_val = len(os.listdir(validation_Functional_dir))
    num_Microcrack_val = len(os.listdir(validation_Microcrack_dir))
    num_Darkspot_val = len(os.listdir(validation_Darkspot_dir))

    total_train = num_Fingerinteruptions_tr + num_Microcrack_tr + num_Functional_tr + num_Cellfaults_tr + num_Break_tr + num_Materialdefect_tr + num_Darkspot_tr
    total_val = num_Fingerinteruptions_val + num_Microcrack_val + num_Functional_val + num_Cellfaults_val + num_Break_val + num_Materialdefect_val + num_Darkspot_val 
    print('total training Finger-interuptions images:', num_Fingerinteruptions_tr)
    print('total training Material-defect images:', num_Materialdefect_tr)
    print('total training Break images:', num_Break_tr)
    print('total training Cell-faults images:', num_Cellfaults_tr)
    print('total training Functional images:', num_Functional_tr)
    print('total training Microcrack images:', num_Microcrack_tr)
    print('total training Dark-spot images:', num_Darkspot_tr)

    print('total validation Finger-interuptions images:', num_Fingerinteruptions_val)
    print('total validation Material-defect images:', num_Materialdefect_val)
    print('total validation Break images:', num_Break_val)
    print('total validation Cell-faults images:', num_Cellfaults_val)
    print('total validation Functional images:', num_Functional_val)
    print('total validation Microcrack images:', num_Microcrack_val)
    print('total validation Dark-spot images:', num_Darkspot_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)
    return total_train,total_val

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), padding='same',input_shape=(224, 224, 3),kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(8, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(16, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    #model.add(layers.Conv2D(16, (3, 3)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    #model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model



# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_results(history,epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0])
    plt.grid()  # 生成网格
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid()  # 生成网格

    plt.show()




train_dir = os.path.join(r'E:\Application\elpv-dataset\newData_r\train')
validation_dir = os.path.join(r'E:\Application\elpv-dataset\newData_r\test')
total_train, total_val = getInfo(train_dir, validation_dir) 

batch_size = 32
epochs = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
print(train_data_gen.class_indices)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

#sample_training_images, _ = next(train_data_gen)
#plotImages(sample_training_images[:5])


model = build_model()    # Build the CNN model and determine the loss function and the optimizer
     # Start the training
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size)


     # Save model or save the weight
#model.save('E:\\Application\\elpv-dataset\\my_model.h5')
model.save_weights('E:\\Application\\elpv-dataset\\weights.h5')
print("Saved as 'weights.h5'")

plot_results(history,epochs)



