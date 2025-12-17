# Program to train the Keras ConvNet and use it for predictions


############ Import all the necessary modules #############

from __future__ import division, print_function
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import scipy.signal as signal
import time
import os, sys, argparse
import dataset2
import matplotlib.pyplot as plt
import pandas as pd
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score, recall_score,f1_score
#from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from rtlsdr import RtlSdr
import pickle


####### Plot accuracy and loss vs epoch ###############

# Function to plot history for accuracy

#def plot_accuracy_vs_epoch(history):
#    if opt=='1':
#        plt.plot(history['accuracy'])
#        plt.plot(history['val_accuracy'])
#    else:
#        plt.plot(history['accuracy'])
#        plt.plot(history['val_accuracy'])        
#    plt.title('Model Accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.savefig('Accuracy.png')
#    plt.close()

# Function to plot history for loss
#def plot_loss_vs_epoch(history):
#    if opt=='1':
#        plt.plot(history['loss'])
#        plt.plot(history['val_loss'])
#    else:
#        plt.plot(history['loss'])
#        plt.plot(history['val_loss'])        
#    plt.title('Model Loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper right')
#    plt.savefig('Loss.png')
#    plt.close()

############ Visualizing filters and feature maps ##########

# Summarize filter shapes
#def plot_filters():
#    print("Filter shapes\n")
#    for layer in model.layers:
#        # Check for convolutional layer
#        if 'conv' not in layer.name:
#            continue
#        # Get filter weights
#        filters, biases = layer.get_weights()
#        print(layer.name, filters.shape)
#        k=1
#        # Gormalize filter values to 0-1 so we can visualize them
#        f_min, f_max = filters.min(), filters.max()
#        filters = (filters - f_min) / (f_max - f_min)
#        # Plot filters
#        n_filters, ix = filters.shape[3], 1
#        print(n_filters,"number of filters in this layer")
#    
#        fig = plt.figure()
#        for i in range(n_filters):
#            # Get the filter
#            f = filters[:, :, :, i]
#            # Add all channels
#            f = np.sum(f,axis=2,keepdims=True)
#            # Specify subplot and turn of axis
#            ax = fig.add_subplot(8, n_filters/8, ix)
#            # Plot filter channel in grayscale
#            plt.imshow(f[:, :,0], cmap='gray')
#            ax.set_xticks([])
#            ax.set_yticks([])
#            ix += 1  
#        plt.savefig("Layer" + str(k) + "filters.png")
#        k=k+1
#        print("Done saving filters image")
#        plt.close()
############ Feature Maps ###########################
#
#def plot_feature_maps(layer_index):
#    print(layer_index)
#    model1 = Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
#    #model1.summary()
#    # convert the image to an array
#    img = Xtrain[0,:,:,:]
#    # expand dimensions so that it represents a single 'sample'
#    img = np.expand_dims(img, axis=0)
#    # get feature map for first hidden layer
#    feature_maps = model1.predict(img)
#    # plot all 64 maps in an 8x8 squares
#    n_col = 8
#    n_row = int(feature_maps.shape[3]/n_col)
#    ix = 1
#    for _ in range(n_row):
#        for _ in range(n_col):
#            # specify subplot and turn of axis
#            ax = plt.subplot(n_row, n_col, ix)
#            ax.set_xticks([])
#            ax.set_yticks([])
#            # plot filter channel in grayscale
#            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
#            ix += 1
#    # show the figure
#    plt.savefig("Featuremap-Layer_" + str(layer_index) +".png")
#    plt.close()

############## Plotting Confusion Matrix ########################

#def plot_cm(test_path,plot_extn):
#    #test_path = 'testing_data'  # Folder for the data used for testing
#    data = dataset2.read_train_sets2(test_path, classes, validation_size=0) # Gets the data object using a class in dataset2.py
#    
#    Xtest1 = data.train.images # Testing samples
#    Ytest1 = data.train.labels # Testing labels
#    print("Testing data prep done")
#    results = model.evaluate(Xtest1, Ytest1, batch_size=128)
#    print("tset.loss,test acc:",results)
#    Ypred1 = model.predict(Xtest1) # Use model to predict
#    # Ypred1 is in SoftMax form, convert to One-Hot Encoded
#    index = np.argmax(Ypred1,axis=1) # Get index of maximum element of each testing example
#    for i in range(Ypred1.shape[0]):
#        for j in range(Ypred1.shape[1]):
#            if j == index[i]:
#                Ypred1[i][j] = 1
#            else:
#                Ypred1[i][j] = 0
#            
#    y_test = pd.Series(np.argmax(Ytest1,axis=1),name = 'Actual')    # Form dataframe of actual labels    
#    y_pred = pd.Series(np.argmax(Ypred1,axis=1),name = 'Predicted') # Form dataframe of predicted labels
#    df_confusion = pd.crosstab(y_test, y_pred) # Crosstab to get in required format
#    df_conf_norm = df_confusion / df_confusion.sum(axis=1) # Normalise it
##    sn.heatmap(df_conf_norm, annot=True, annot_kws={"size": 8}, cmap="Green")
#    plt.matshow(df_conf_norm, cmap=plt.cm.gray_r) # Plot figure
#    plt.colorbar() # Colourbar
#    ax = plt.gca()
#    #plt.locator_params(numticks=10)
#    #plt.xticks(np.arange(0,9,step=1))
#    #ax.set_yticklabels([" ", "GFSK", "32PSK", "OOK", "GMSK", "16PSK", "16QAM", "CPFSK", "BPSK", "8PSK", "QPSK", "64QAM"])
#    #ax.set_xticklabels([" ", "GFSK", "32PSK", "OOK", "GMSK", "16PSK", "16QAM", "CPFSK", "BPSK", "8PSK", "QPSK", "64QAM"])
#    labels = classes # ["GFSK", "32PSK", "OOK", "GMSK", "16PSK", "16QAM", "CPFSK", "BPSK", "8PSK", "QPSK"]    
#    plt.xticks(np.arange(0,10),labels,rotation='vertical')
#    plt.yticks(np.arange(0,10),labels)
#    plt.ylabel(df_confusion.index.name)
#    plt.xlabel(df_confusion.columns.name)
#    plt.savefig(plot_extn+"ConfusionMatrix.png")
#    plt.close()
#    cm = ConfusionMatrix(y_test, y_pred)
#    #cm.print_stats() # it gives the confusion matrices
#    #print(classification_report(y_test, y_pred)) #it display the precision,accuracy and fscore
#    #print(classes)
#    return cm, classification_report(y_test, y_pred)


############ Define some 0f the functions #####################

#Parser function that handles parsing of command line arguments

def build_parser():
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-decimation_rate', dest = 'decimation_rate', type = int, 
         default = 12, help = 'Decimation rate of the signal')
    parser.add_argument('-sampling_rate', dest = 'sampling_rate', type = int, 
         default = 2400000, help = 'Sampling rate of the signal')
    parser.add_argument('-sdr', dest = 'sdr', type = int, 
         default = 1, help = 'Read samples from file (0) or device (1)')
    return parser



def prepare_args():
    # hack, http://stackoverflow.com/questions/9025204/
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg
    parser = build_parser()
    args = parser.parse_args()
    return args


# Function to read the samples from _prediction_samples.dat files

def read_samples_sdr(freq):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.err_ppm = 56   # change it to yours
    sdr.gain = 'auto'

    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


def read_samples(freq):
    f_offset = 250000  # Shifted tune to avoid DC
    samp = np.fromfile(str(freq)+'_prediction_samples.dat',np.uint8)+np.int8(-127) # Adding a signed int8 to an unsigned one results in an int16 array
    x1 = samp[::2]/128 # Even samples are real(In-phase)
    x2 = samp[1::2]/128 # Odd samples are imaginary(Quadrature-phase)
    iq_samples = x1+x2*1j # Create the complex data samples
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # Shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples

# Function that checks the model's prediction with known class
    
#def check(freq, corr):
#    samples = []
#    if args.sdr == 1:
#            iq_samples = read_samples_sdr(freq)
#    elif args.sdr == 0:
#        iq_samples = read_samples(freq)
#    iq_samples = signal.decimate(iq_samples, args.decimation_rate, zero_phase=True) # Decimate the signal according to the required factor
#
#    real = np.real(iq_samples)
#    imag = np.imag(iq_samples)
#
#    iq_samples = np.ravel(np.column_stack((real, imag))) # Store both real and imaginary data in one row
#    iq_samples = iq_samples[:INPUT_DIM] # Shaping data to required input dimensions
#
#    samples.append(iq_samples)
#    samples = np.array(samples)
#    samples = np.reshape(samples, (len(samples), DIM1, DIM2, 2)) # Reshape for convolutional model
#
#    prediction = model.predict(samples) # Use the trained model to predict signal classes
#
#    # Print predicted label
#    maxim = 0.0
#    maxlabel = ""
#    for sigtype, probability in zip(classes, prediction[0]):
#        if probability >= maxim:
#            maxim = probability
#            maxlabel = sigtype
#    print(freq / 1000000, maxlabel, maxim * 100)
#    print('\n')
#    print(classes)
#    print(prediction[0])
#    # Calculate validation percent
#    if corr == maxlabel:
#        global correct_predictions
#        correct_predictions += 1





########### Main Code #################################
#args = prepare_args()  # Get the decimation rate from command line
#sample_rate = args.sampling_rate
#correct_predictions = 0
#opt=input('Enter Option (1: Train & Save the model, 2: Test):')#1
############ Data for training the model #################
scenario = 'high'
train_path = 'training_data'  # Folder for the data used for training
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))] # Gets classes from all the folders mentioned under training_data folder
num_classes = len(classes)
data = dataset2.read_train_sets2(train_path, classes, validation_size=0.3) # Gets the data object using a class in dataset2.py


Xtrain = data.train.images
Ytrain = data.train.labels
Xtest = data.valid.images
Ytest = data.valid.labels

######### Convolutional Neural Network Architecture ######

#if opt=='1':
    # Dimensions of input data tensor
print("Model building beginning") 
DIM1 = 28
DIM2 = 28
INPUT_DIM = 1568
    
input_signal = Input(shape=(DIM1, DIM2, 2)) # Input tensor
x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_signal) # First convolutional layer
#x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x) # Maxpooling
#x = BatchNormalization()(x)
#x = Dropout(0.1)(x)
x = Conv2D(64,(3, 3), activation='relu', padding='same')(x) # Second convolutional layer
#x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x) # Maxpooling
#x = BatchNormalization()(x)  
#x = Dropout(0.1)(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) # Third convolutional layer
#x = BatchNormalization()(x) 
#x = MaxPooling2D((2, 2), padding='same')(x) # Maxpooling
#x = BatchNormalization()(x)
#x = Dropout(0.1)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 4th convolutional layer
#x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x) # Maxpooling
#x = BatchNormalization()(x)
#x = Dropout(0.2)(x)
x = Flatten()(x) # Flatten the layer 
# #   x = Dropout(0.051)(x)
x = Dense(128, activation='relu')(x) # Dense layer
#x = BatchNormalization()(x) 
#x = Dropout(0.1)(x)
x = Dense(num_classes, activation='softmax')(x) # Dense layer with SoftMax activation for output
    
    ######### Training the model using Keras Utils ###########
    
model = Model(inputs=input_signal, outputs=x) # Creates a model object of the Keras Model class with required input and ConvNet architecture dimesnions
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Specify the loss function type, optimiser type and metrics of the ConvNet
history = model.fit(Xtrain, Ytrain, epochs=70, batch_size=128, shuffle=True, validation_data=(Xtest, Ytest)) # Train the ConvNet
    
model.save(train_path+'SNR_Model_eight_24')
pickle.dump(history.history,open(train_path+'SNR_history_eight_24','wb'))

#elif opt=='2':
#    model = keras.models.load_model(scenario+'SNR_Model')
#    history = pickle.load(open(scenario+'SNR_history','rb'))

#print("Model Fit done")
#print("Model Summary: ")
#model.summary() # Summarize the model

########### Plot the Model Accuracy and Loss wrt epoch ######

# list all data in history
#if opt==1:
#    print(history.history.keys())
#else:
#    print(history.keys())

#    
#print('Plotting  accuracy and loss vs epoch')
#plot_accuracy_vs_epoch(history)
#plot_loss_vs_epoch(history)
#
#plot_filters()

#layer_index = 1
#plot_feature_maps(layer_index)
#layer_index = 3
#plot_feature_maps(layer_index)
#layer_index = 5
#plot_feature_maps(layer_index)


#print('Plotting Confusion Matrix')
#test_path = '/mnt/20528F7C528F5586/CNN/'+scenario+'SNR/testing_data'  # Folder for the data used for testing
#plot_extn = 'medium'
#conf_matrix,classification_report = plot_cm(test_path,plot_extn)
#print(conf_matrix.print_stats())
#print('Classification Report:')
#print(classification_report)
#print('Checking')        

#for i in range(1,100):
#    x=input('Enter the class name :')
#    f=810000000 #input('Enter frequency:')
#    check(f,x)

#print("Validation:", correct_predictions / 10 * 100)
