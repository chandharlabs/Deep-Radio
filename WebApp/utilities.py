import argparse
import time
import os, sys
import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr
# from tensorflow import keras
import tflite_runtime.interpreter as tflite
from math import floor


# Data Generation
def build_parser():
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-decimation_rate', dest='decimation_rate', type=int, 
         default=12, help='Decimation rate of the signal')
    parser.add_argument('-sampling_rate', dest='sampling_rate', type=int, 
         default=2400000, help='Sampling rate of the signal')
    parser.add_argument('-sdr', dest='sdr', type=int, 
         default=1, help='Read samples from file (0) or device (1)')
    return parser

def prepare_args():
    # hack, http://stackoverflow.com/questions/9025204/
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg
    parser = build_parser()
    args = parser.parse_args()
    return args

def read_samples_sdr(freq, sample_rate):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.err_ppm = 56  # change it to yours
    sdr.gain = 'auto'

    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples

def read_samples(freq, sample_rate):
    f_offset = 250000  # Shifted tune to avoid DC
    samp = np.fromfile(str(freq)+'_prediction_samples.dat',np.uint8)+np.int8(-127)  # Adding a signed int8 to an unsigned one results in an int16 array
    x1 = samp[::2]/128  # Even samples are real (In-phase)
    x2 = samp[1::2]/128  # Odd samples are imaginary (Quadrature-phase)
    iq_samples = x1+x2*1j  # Create the complex data samples
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # Shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


# Training








# Testing





# Prediction
def check(freq, sample_rate, decimation_rate, sdr, classes, model_path):
    #samples = []
    #DIM1 = 28
    #DIM2 = 28
    #INPUT_DIM = 1568
    #if sdr == 1:
    #    iq_samples = read_samples_sdr(freq, sample_rate)
    #elif sdr == 0:
    #    iq_samples = read_samples(freq, sample_rate)
    #iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True)  # Decimate the signal according to the required factor

    #real = np.real(iq_samples)
    #imag = np.imag(iq_samples)

    #iq_samples = np.ravel(np.column_stack((real, imag)))  # Store both real and imaginary data in one row
    #iq_samples = iq_samples[:INPUT_DIM]  # Shaping data to required input dimensions

    #samples.append(iq_samples)
    #samples = np.array(samples)
    #samples = np.reshape(samples, (len(samples), DIM1, DIM2, 2))  # Reshape for convolutional model

    #prediction = model.predict(samples)  # Use the trained model to predict signal classes

    # Print predicted label
    #maxim = 0.0
    #maxlabel = ""
    #for sigtype, probability in zip(classes, prediction[0]):
    #    if probability >= maxim:
    #        maxim = probability
    #        maxlabel = sigtype
    #return freq / 1000000, maxlabel, maxim * 100

    samples = []
    DIM1, DIM2, INPUT_DIM = 28, 28, 1568
    iq_samples = read_samples_sdr(freq, sample_rate) if sdr == 1 else read_samples(freq, sample_rate)
    iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True)
    real, imag = np.real(iq_samples), np.imag(iq_samples)
    iq_samples = np.ravel(np.column_stack((real, imag)))
    iq_samples = iq_samples[:INPUT_DIM]
    samples.append(iq_samples)
    samples = np.array(samples).reshape((len(samples), DIM1, DIM2, 2)).astype(np.float32)

    # Load TFLite model
    interpreter = tflite.Interpreter('/home/chandharlabs/CNN/highSNR_Model.tflite') #tflite.Interpreter(model_path + model_name) # tflite.Interpreter(model_path + 'highSNR_Model.tflite')

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], samples)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    print(prediction)
    max_idx = np.argmax(prediction)
    print(freq / 1e6, classes[max_idx], prediction[max_idx] * 100)
    return freq / 1e6, classes[max_idx], prediction[max_idx] * 100