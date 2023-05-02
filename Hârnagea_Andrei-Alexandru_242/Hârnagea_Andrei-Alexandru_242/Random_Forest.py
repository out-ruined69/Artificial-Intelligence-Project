# CITIRE DATE
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#Declar modeleul, cu 50 de arbori
rand_for = RandomForestClassifier(n_estimators = 50,max_features = 0.5,
                                  max_depth = 10, min_samples_split = 2,
                                  min_samples_leaf=1,n_jobs=-1)

rand_for = rand_for.fit(poze_train, list_class_train)
y_pred = rand_for.predict(poze_sample)