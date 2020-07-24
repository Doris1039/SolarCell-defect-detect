
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import cv2
import numpy as np



rootDir=r"E:\Application\elpv-dataset\newData_dVSf_r\test\Defective"
os.chdir(rootDir) 
dirs = os.listdir(rootDir)
n = 0
N = 0

source_path = rootDir + "\\"
        
for file in dirs:
        img = load_img(source_path + file)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        '''
        datagen = ImageDataGenerator(
                                rotation_range=2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest',
                                brightness_range = [0.5,0.9])


        '''                        
        datagen = ImageDataGenerator(
                                rotation_range=2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest')
        
        
        it = datagen.flow(samples, 
                        batch_size=2)

        for i in range(5):
                save = source_path  + '\\'  + 'A'+str(i)+'_'+file 

                #pyplot.subplot(330 + 1 + i)
                
                batch = it.next()
                cv2.imwrite(save, batch[0])


                image = batch[0].astype('uint32')
                #pyplot.imshow(image)
        #pyplot.show()
