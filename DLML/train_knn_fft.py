############ FFT (Welch PSD) Based KNN Training #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time
import os, sys, glob
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

t1 = time.time()
directory = os.getcwd()

# ============================================================
# Welch PSD feature extraction configuration
# ============================================================
FEATURE_CFG = {
    "nperseg": 4096,       # FFT window length
    "noverlap": 2048,      # 50% overlap
    "window": "hann",
    "scaling": "density",
    "return_onesided": True,
    "n_features": 1024,    # final resampled length
    "eps": 1e-12,          # numerical stability
}

# ============================================================
# Welch PSD Feature Extractor
# ============================================================
def compute_psd_features(iq_samples, cfg=FEATURE_CFG):
    """
    Compute Welch PSD (log power) features and resample to fixed length.
      1) Welch PSD
      2) 10*log10(PSD + eps)
      3) Resample to cfg['n_features']
      4) Per-sample min-max normalization
    """
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
    if Pxx_dB.size != cfg["n_features"]:
        feat = signal.resample(Pxx_dB, cfg["n_features"])
    else:
        feat = Pxx_dB

    # normalize 0–1
    fmin, fmax = feat.min(), feat.max()
    if fmax - fmin < 1e-9:
        feat_norm = np.zeros_like(feat, dtype=np.float32)
    else:
        feat_norm = ((feat - fmin) / (fmax - fmin)).astype(np.float32)
    return feat_norm

# ============================================================
# Load training samples from folders (FFT features)
# ============================================================
def load_train2(train_path, classes):
    samples, labels, sample_names, cls = [], [], [], []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*.npy')
        files = glob.glob(path)

        for fl in files:
            try:
                iq_samples = np.load(fl)
                feat = compute_psd_features(iq_samples)   # <-- FFT feature extraction
                samples.append(feat)
                labels.append(index)
                sample_names.append(os.path.basename(fl))
                cls.append(fields)
            except Exception as e:
                print(f"Skipping {fl} due to error: {e}")
                continue

    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels, np.array(sample_names), np.array(cls)

# ============================================================
# Data container
# ============================================================
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
    def images(self): return self._images
    @property
    def labels(self): return self._labels
    @property
    def img_names(self): return self._img_names
    @property
    def cls(self): return self._cls

# ============================================================
# Create training + validation splits
# ============================================================
def read_train_sets2(train_path, classes, validation_size):
    class DataSets(object): pass
    data_sets = DataSets()

    images, labels, img_names, cls = load_train2(train_path, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=42)

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

# ============================================================
# MAIN TRAINING
# ============================================================
training_data_folder = 'training_data'
class_array = ['BPSK', 'QPSK', 'GMSK']

train_path = os.path.join(directory, training_data_folder)
num_classes = len(class_array)

data = read_train_sets2(train_path, class_array, validation_size=0.3)

Xtrain = data.train.images
Ytrain = data.train.labels
Xtest = data.valid.images
Ytest = data.valid.labels

print('Training dataset shape:', Xtrain.shape)
print('Training labels shape:', Ytrain.shape)
print('Validation dataset shape:', Xtest.shape)
print('Validation labels shape:', Ytest.shape)

# ============================================================
# Train KNN
# ============================================================
print("\nTraining KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
model = knn.fit(Xtrain, Ytrain)

# ============================================================
# Evaluation
# ============================================================
y_pred = model.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)

print("\nAccuracy: {:.4f}".format(acc))
print("\nClassification Report:\n", classification_report(Ytest, y_pred, target_names=class_array))
print("\nConfusion Matrix:\n", confusion_matrix(Ytest, y_pred))

# ============================================================
# Save Model
# ============================================================
pickle.dump(
    {"model": model, "feature_cfg": FEATURE_CFG, "class_array": class_array},
    open('model_knn_welchpsd.sav', 'wb')
)

print("\nSaved model: model_knn_welchpsd.sav")
print("Time taken: {:.2f} s".format(time.time() - t1))

