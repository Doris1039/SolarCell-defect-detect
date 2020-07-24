import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import datasets, layers, models
import os
from PIL import Image


def load_dataset(fname=None):
    if fname is None:
        # Assume we are in the utils folder and get the absolute path to the
        # parent directory.
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir))
        fname = os.path.join(fname, "E:\\Application\\elpv-dataset\\data_test\\labels_eval_dn.csv")

    data = np.genfromtxt(fname, delimiter=' ', dtype=['|S100', '|S20', '<i8'], names=[
                         'path', 'DefectType', 'Num'])
    image_fnames = np.char.decode(data['path'])
    Num = data['Num']
    #image_name = np.char.decode(data['path'])

    DefectType = np.char.decode(data['DefectType'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    dir = os.path.dirname(fname)

    images = np.array([load_cell_image(os.path.join(dir, fn))
                       for fn in image_fnames])

    return images, DefectType, Num

def build_model():
    
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), input_shape=(224, 224, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(8, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(16, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #model.add(layers.Conv2D(16, (3, 3)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def plot_image(i, predictions_array, true_label, img):
  class_names = ['Break', 'Cell-faults', 'Dark-spot', 'Finger-interuptions', 
                                        'Functional', 'Material-defect', 'Microcrack']
  #class_names = ['Defective', 'Functional']  
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'


  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

                       

model = build_model()
model.load_weights('E:\\Application\\elpv-dataset\\weights.h5')

#model = load_model('E:\\Application\\elpv-dataset\\my_model.h5')  

print('Loading...')

test_images,  test_type, test_labels = load_dataset()
#test_img=tf.keras.preprocessing.image.load_img(filename,target_size=(224,224,3))
#test_img=tf.keras.preprocessing.image.img_to_array(test_img)
test_img=test_images/255  
print('Predicting...')

#test_img=np.expand_dims(test_img,axis=0)
predictions = model.predict(test_img)


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 4
num_cols = 3
num_images = num_rows*num_cols
for n in range(78):
  for i in range(num_images):
    Num = i + n * 12
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(Num, predictions[Num], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(Num, predictions[Num], test_labels)
  #plt.tight_layout()
  plt.show()

