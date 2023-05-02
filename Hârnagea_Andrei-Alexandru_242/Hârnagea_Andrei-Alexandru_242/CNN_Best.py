import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn import preprocessing

# matplotlib.pyplot as plt

# citire labeluri
f = open("/kaggle/input/unibuc-brain-ad/data/train_labels.txt", "r")
f.readline()

list_id_train = []
list_class_train = []

for l in f.readlines():
    a, b = l.split(',')
    list_id_train.append(a)
    list_class_train.append(b[0])

# print(list_class_train)

f = open("/kaggle/input/unibuc-brain-ad/data/sample_submission.txt", 'r')
f.readline()
list_id_sample = []
# list_class_sample = []

for l in f.readlines():
    a, b = l.split(',')
    list_id_sample.append(a)
    # list_class_sample.append(b[0])

# print(list_id_sample)
f.close()

f = open("/kaggle/input/unibuc-brain-ad/data/validation_labels.txt", "r")
f.readline()
list_id_validation = []
list_class_validation = []

for l in f.readlines():
    a, b = l.split(',')
    list_id_validation.append(a)
    list_class_validation.append(b[0])

#citire poze
poze_train = []
for index in list_id_train:
    image = cv2.imread(f'/kaggle/input/unibuc-brain-ad/data/data/' + index + '.png', cv2.IMREAD_GRAYSCALE) #le fac gray si le dau flatten
    poze_train.append((image).flatten())
poze_train = np.array(poze_train)

poze_sample = []
for index in list_id_sample:
    image = cv2.imread(f'/kaggle/input/unibuc-brain-ad/data/data/{index}.png', cv2.IMREAD_GRAYSCALE) #le fac gray si le dau flatten
    poze_sample.append((image).flatten())
poze_sample = np.array(poze_sample)

poze_validation = []
for index in list_id_validation:
    image = cv2.imread(f'/kaggle/input/unibuc-brain-ad/data/data/{index}.png', cv2.IMREAD_GRAYSCALE) #le fac gray si le dau flatten
    poze_validation.append((image).flatten())
poze_validation = np.array(poze_validation)

list_class_train = np.array(list_class_train, dtype=np.float32)
list_class_validation = np.array(list_class_validation, dtype=np.float32)
#fac clasele int-uri

#Am incercat sa fac si un data generator, dar am dat de probleme
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
# )
#
# poze_augumentate = []
# labels_augumentate = []
#
# for i in range(len(poze_train)):
#     image = poze_train[i].reshape((224, 224, 1))
#     label = list_class_train[i]
#
#     for j in range(2):
#         poza_aug = datagen.random_transform(image)
#         poze_augumentate.append(poza_aug.flatten())
#         labels_augumentate.append(label)
#
# poze_train = np.concatenate((poze_train, poze_augumentate))
# list_class_train = np.concatenate((list_class_train, labels_augumentate))

from sklearn.metrics import f1_score

poze_train = poze_train.reshape(-1, 64, 64, 1)
poze_validation = poze_validation.reshape(-1, 64, 64, 1)
poze_sample = poze_sample.reshape(-1, 64, 64, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.layers import BatchNormalization
import tensorflow as tf

batch_size = 64

#Aici creez modelul efectiv
model = Sequential()

model.add(Conv2D(256,(3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPool2D(2,2))


model.add(Conv2D(64,(3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())


model.add(Conv2D(128,(3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())


model.add(Conv2D(64,(3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())

#4 layere asemanatorea, schimb numarul de features. dupa fiecare layer am BatchNormalization.

model.add(Flatten())

#dau flatten inputului

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2)) #pentru a nu face overitting
model.add(Dense(1, activation = 'sigmoid'))

#functia sigmoid permite clasificarea binara

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#compilez modelul

history = model.fit(poze_train,list_class_train, epochs = 15, batch_size= batch_size, validation_data = (poze_validation,list_class_validation))
#fac o predictie, pot schimba numarul de epoci

y_pred = model.predict(poze_validation)

file = open("/kaggle/working/CNN_PRED.csv", 'w')
file.write("id,class")
file.write('\n')
for i in range(len(y_pred)):
    file.write(list_id_sample[i])
    file.write(',')
    if(y_pred[i] < 0.5):
        file.write('0')
    else:
        file.write('1')
    file.write('\n')