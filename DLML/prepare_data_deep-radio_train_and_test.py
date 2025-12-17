# File that prepares training and testing files
# Authors
# Vinay  U. Pai  (email: f20170131@pilani.bits-pilani.ac.in)

import argparse, time, random, string, os, sys
import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr

    
def read_samples_sdr(freq):
    sdr = RtlSdr()
    sdr.sample_rate = sampling_rate
    sdr.err_ppm = 56   # change it to yours
    sdr.gain = 40.2 #"auto"

    f_offset = 250000 # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sampling_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def collect_samples(freq, classname, decimation_rate):
    os.makedirs("training_data/" + classname, exist_ok=True)
    os.makedirs("testing_data/" + classname, exist_ok=True)
    for i in range(0, 500):
        iq_samples = read_samples_sdr(freq)
        iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True)
        if (i < 375):  # 75% train, 25% test
            filename = "training_data/" + classname + "/samples-" + randomword(16) + ".npy"
        else:
            filename = "testing_data/" + classname + "/samples-" + randomword(16) + ".npy"
        np.save(filename, iq_samples)
        if not (i % 5): print(i / 5, "%", classname)

decimation_rate = 12
sampling_rate = 2400000

for i in range (1,3):
    x=input('Enter class name:')
    collect_samples(700000000,x,decimation_rate)
