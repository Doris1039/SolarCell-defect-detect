import tensorflow as tf 
import numpy as np
from tensorflow.keras.preprocessing import image

model=tf.keras.models.load_model('E:\\elpv-dataset\\my_model.h5')

file_path=r"E:\elpv-dataset\img\dd\cell0008.png" ##随便啥图都可以.
img=image.load_img(file_path,target_size=(200,200))
img = np.expand_dims(image, axis=0)
x=image.img_to_array(img)

y=model.predict(x)

print(y)
# 重新创建完全相同的模型，包括其权重和优化程序



# 显示网络结构
model.summary()
