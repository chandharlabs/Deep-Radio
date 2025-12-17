############ Import all the necessary modules #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time
import os, sys, argparse, glob
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
t1=time.time()
import pickle

directory = os.getcwd()


def load_train2(train_path, classes):
    samples = []
    labels = []
    sample_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*.npy')
        files = glob.glob(path)
        for fl in files:
            iq_samples = np.load(fl)
            real = np.real(iq_samples)
            imag = np.imag(iq_samples)

            iq_samples = np.ravel(np.column_stack((real, imag)))

            samples.append(iq_samples)

            label=index
            labels.append(label)
            flbase = os.path.basename(fl)
            sample_names.append(flbase)
            cls.append(fields)

    samples = np.array(samples)
    labels = np.array(labels)
    sample_names = np.array(sample_names)
    cls = np.array(cls)
    return samples, labels, sample_names, cls


# Define the data class

class DataSet2(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets2(train_path, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train2(train_path, classes)  # 2 calculating, 3 loading
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
#    print(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet2(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet2(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets

def read_samples_fc32(filename):
    iq_samples = np.fromfile(filename, np.complex64)# + np.int8(-127) #adding a signed int8 to an unsigned one results in an int16 array
    return iq_samples/64

   
def check(filename,model,classes):
    samples = []
    decimation_rate=1
    iq_samples =np.load(filename)
    iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True) # Decimate the
    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    iq_samples = np.ravel(np.column_stack((real, imag))) # Store both real and imaginary data in one row


    samples.append(iq_samples)
    samples = np.array(samples)
    
    prediction = model.predict(samples) # Use the trained model to predict signal classes

    print(classes[prediction[0]])

########### Main Code #################################

class_array=['BPSK','QPSK','GMSK']

#    ###################################### 'predict':


model=pickle.load(open('model_knn.sav', 'rb'))
sample_rate = 2.4e6
freq = 957e6
check(freq,model,class_array)

