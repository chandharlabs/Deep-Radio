############ FFT (Welch PSD) Based SVM Training #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time, os, glob, pickle
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

t1 = time.time()
directory = os.getcwd()

# =========================
# Welch PSD feature config
# =========================
FEATURE_CFG = {
    "nperseg": 4096,
    "noverlap": 2048,
    "window": "hann",
    "scaling": "density",
    "return_onesided": True,
    "n_features": 1024,   # final feature length
    "eps": 1e-12,
}

def compute_psd_features(iq_samples, cfg=FEATURE_CFG):
    """Welch PSD -> log10 power -> resample to fixed length -> per-sample min-max [0,1]."""
    f, Pxx = signal.welch(
        x=iq_samples,
        window=cfg["window"],
        nperseg=cfg["nperseg"],
        noverlap=cfg["noverlap"],
        return_onesided=cfg["return_onesided"],
        scaling=cfg["scaling"],
        detrend=False
    )
    Pxx_dB = 10.0 * np.log10(Pxx + cfg["eps"])
    feat = signal.resample(Pxx_dB, cfg["n_features"]) if Pxx_dB.size != cfg["n_features"] else Pxx_dB

    fmin, fmax = feat.min(), feat.max()
    if fmax - fmin < 1e-9:
        return np.zeros_like(feat, dtype=np.float32)
    return ((feat - fmin) / (fmax - fmin)).astype(np.float32)

def load_train2(train_path, classes):
    samples, labels, sample_names, cls = [], [], [], []
    for fields in classes:
        idx = classes.index(fields)
        for fl in glob.glob(os.path.join(train_path, fields, '*.npy')):
            try:
                iq = np.load(fl)
                feat = compute_psd_features(iq)   # <-- FFT features (NO decimation)
                samples.append(feat)
                labels.append(idx)
                sample_names.append(os.path.basename(fl))
                cls.append(fields)
            except Exception as e:
                print(f"Skipping {fl} due to error: {e}")
                continue
    return np.array(samples), np.array(labels), np.array(sample_names), np.array(cls)

class DataSet2(object):
    def __init__(self, images, labels, img_names, cls):
        self._images, self._labels = images, labels
        self._img_names, self._cls = img_names, cls
        self._num_examples = images.shape[0]

def read_train_sets2(train_path, classes, validation_size):
    class DataSets(object): pass
    data_sets = DataSets()
    images, labels, img_names, cls = load_train2(train_path, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=42)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    data_sets.valid = DataSet2(images[:validation_size], labels[:validation_size],
                               img_names[:validation_size], cls[:validation_size])
    data_sets.train = DataSet2(images[validation_size:], labels[validation_size:],
                               img_names[validation_size:], cls[validation_size:])
    return data_sets

########### Main Code #################################
training_data_folder = 'training_data'
class_array = ['BPSK','QPSK','GMSK']

train_path = os.path.join(directory, training_data_folder)
data = read_train_sets2(train_path, class_array, validation_size=0.3)

Xtrain, Ytrain = data.train._images, data.train._labels
Xtest,  Ytest  = data.valid._images, data.valid._labels

print('Training dataset shape:', Xtrain.shape)
print('Training labels shape:',  Ytrain.shape)
print('Validation dataset shape:', Xtest.shape)
print('Validation labels shape:',  Ytest.shape)

# SVM (linear kernel). probability=True enables calibrated probabilities (Platt scaling).
svc = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
model = svc.fit(Xtrain, Ytrain)

y_pred = model.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
print("\nAccuracy: {:.4f}".format(acc))
print("\nClassification Report:\n", classification_report(Ytest, y_pred, target_names=class_array))
print("\nConfusion Matrix:\n", confusion_matrix(Ytest, y_pred))

# Save model + feature cfg + class order
pickle.dump(
    {"model": model, "feature_cfg": FEATURE_CFG, "class_array": class_array},
    open('model_svm_welchpsd.sav', 'wb')
)

print("\nSaved model: model_svm_welchpsd.sav")
print("Time taken: {:.2f} s".format(time.time() - t1))

