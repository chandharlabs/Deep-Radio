############ Train Naive Bayes (No Decimation, Welch PSD 1024) ############

import numpy as np, scipy.signal as signal, os, glob, pickle, time
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

t1 = time.time()
directory = os.getcwd()

FEATURE_CFG = {
    "nperseg": 4096, "noverlap": 2048, "window": "hann",
    "scaling": "density", "return_onesided": True,
    "n_features": 1024, "eps": 1e-12,
}

def compute_psd_features(iq, cfg=FEATURE_CFG):
    f, Pxx = signal.welch(iq, window=cfg["window"], nperseg=cfg["nperseg"],
                          noverlap=cfg["noverlap"], scaling=cfg["scaling"],
                          return_onesided=cfg["return_onesided"], detrend=False)
    Pxx_dB = 10*np.log10(Pxx + cfg["eps"])
    feat = signal.resample(Pxx_dB, cfg["n_features"])
    fmin, fmax = feat.min(), feat.max()
    return ((feat - fmin)/(fmax - fmin)).astype(np.float32)

def load_train(train_path, classes):
    X, Y = [], []
    for c in classes:
        idx = classes.index(c)
        for fl in glob.glob(os.path.join(train_path, c, '*.npy')):
            iq = np.load(fl)     # ALREADY DECIMATED — DO NOT DECIMATE AGAIN
            X.append(compute_psd_features(iq))
            Y.append(idx)
    return np.array(X), np.array(Y)

classes = ['BPSK','QPSK','GMSK']
train_path = os.path.join(directory, 'training_data')

print("\nLoading + FFT feature extraction ...")
X, Y = load_train(train_path, classes)
X, Y = shuffle(X, Y, random_state=42)

split = int(0.7*len(X))
Xtrain, Xtest = X[:split], X[split:]
Ytrain, Ytest = Y[:split], Y[split:]

model = GaussianNB().fit(Xtrain, Ytrain)
yp = model.predict(Xtest)
print("\nAccuracy:", accuracy_score(Ytest, yp))
print("\nReport:\n", classification_report(Ytest, yp, target_names=classes))
print("\nConfusion:\n", confusion_matrix(Ytest, yp))

pickle.dump({"model":model,"feature_cfg":FEATURE_CFG,"class_array":classes},
            open('model_nb_welchpsd.sav','wb'))
print("\nModel saved: model_nb_welchpsd.sav")
print("Time taken: %.2fs" % (time.time()-t1))

