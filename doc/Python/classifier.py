# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from keras.utils import np_utils

from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers import Dense, Dropout, Activation         # FNN
from keras.layers import GRU, LSTM, Embedding               # RNN
from keras.layers import Conv2D, Flatten, MaxPooling2D      # CNN
from keras.optimizers import Adam


def load_mfcc():
    # Load data set
    X = np.loadtxt('../data/X_40_long.txt')
    t = pd.read_csv('../data/train_long.csv')
    classes = t.Class.values
    
    # One hot encode
    lb = LabelEncoder()
    encodedClasses = lb.fit_transform(np.unique(classes))
    labelClasses = np.unique(classes)
    codemap = dict(zip(encodedClasses, labelClasses))
    with open('../data/codemap.txt','w') as f:
        f.write(str(codemap))
    t = np_utils.to_categorical(lb.fit_transform(classes))

    # Split into training and test set
    N = len(X)
    N_train = int(N*0.8)
    X_train = X[:N_train]
    t_train = t[:N_train]
    X_val = X[N_train:]
    t_val = t[N_train:]
    return X_train, t_train, X_val, t_val
    
    
    
def load_spectrogram():
    # Load data set
    X = np.loadtxt('../data/spectrogram_40_long.txt')
    t = pd.read_csv('../data/train_long.csv')
    classes = t.Class.values
        
    X = np.reshape(X, (3637, 40, 173, 1)) #3637;136      #CNN needs 4D array as input #(5433, 40, 173, 1)
        
    # One hot encode
    lb = LabelEncoder()
    encodedClasses = lb.fit_transform(np.unique(classes))
    labelClasses = np.unique(classes)
    codemap = dict(zip(encodedClasses, labelClasses))
    t = np_utils.to_categorical(lb.fit_transform(classes))

    # Split into training and test set
    N = len(X)
    N_train = int(N*0.8)
    X_train = X[:N_train]
    t_train = t[:N_train]
    X_val = X[N_train:]
    t_val = t[N_train:]
    return X_train, t_train, X_val, t_val
    


# Run a deep learning model and get results

def Logistic():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(num_labels, input_shape=(40,), W_regularizer=l2(1.0)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X_train, t_train, batch_size=32, epochs=100, validation_data=(X_val, t_val))
    

def FNN(N=1):
    X_train, t_train, X_val, t_val = load_mfcc()

    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(1024, input_shape=(40,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    for i in range(N-1):
        model.add(Dense(1024))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    hist = model.fit(X_train, t_train, batch_size=32, epochs=100, validation_data=(X_val, t_val))
    
    return hist, model
    


def Convolutional():
    X_train, t_train, X_val, t_val = load_spectrogram()
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(40,173,1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.15))
    
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    hist = model.fit(X_train, t_train, batch_size=64, epochs=75, validation_data=(X_val, t_val))
    
    return hist, model
    
def Long_short():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
    X_val = np.reshape(X_val, (len(X_val), len(X_val[0]), 1))
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    #model.add(Embedding(1000, 512, input_length = X_train.shape[1]))
    
    model.add(LSTM(256,input_shape=(40,1),return_sequences=False))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    #model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    hist = model.fit(X_train, t_train, batch_size=32, epochs=100, validation_data=(X_val, t_val))
    
    return hist, model
    
def Gated():
    X_train, t_train, X_val, t_val = load_mfcc()
    
    X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
    X_val = np.reshape(X_val, (len(X_val), len(X_val[0]), 1))
    
    num_labels = t_train.shape[1]

    model = Sequential()
    
    model.add(GRU(256, activation='relu', recurrent_activation='hard_sigmoid'))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    #model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    hist = model.fit(X_train, t_train, batch_size=32, epochs=100, validation_data=(X_val, t_val))
    
    return hist, model

def plot_fitness(hist, output, modelName):
    #generate and save plots-
    # plot history for accuracy
    plt.figure()
    plt.ioff()
    plt.plot(hist.history['accuracy']) #plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy']) #plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(output+'_accuracy_'+modelName, bbox_inches = "tight")
    
    # plot history for loss
    plt.figure()
    plt.ioff()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(output+'_loss_'+modelName, bbox_inches = "tight")

    
if __name__ == '__main__':
    X_train, t_train, X_val, t_val = load_mfcc()
    
    #Logistic()
    #hist, model = FNN(2)
    #hist, model = Convolutional()
    #hist, model = Long_short()
    hist, model = Gated()
        
    # save plots
    output = '../plots/'
    modelName = 'model_gated_long' # WARNING! change if exec other algo
    plot_fitness(hist, output, modelName)
    
    # save model
    output = '../model/'
    model.save(output+modelName+'.h5')
    
    # save detailed model summary    
    with open(output+modelName+".txt","w+") as f:
        
        f.write('Trainparameter:\n')
        #f.write('ValidationSplit: {0}\n'.format(validationSplit))
        #f.write('Batchsize: {0}\n'.format(batchSize))
        #f.write('Epochs: {0}\n'.format(epochs))
        #f.write('Steps per epoch: {0}\n'.format(steps_per_epoch))
        f.write('Inputdimension:\n'+str(np.shape(X_train[0])))
        f.write('\n')
        f.write('Results:\n')
        f.write('Trainaccuracy: {0}\n'.format(hist.history['accuracy'][-1]*100))
        f.write('Validationaccuracy: {0}\n'.format(hist.history['val_accuracy'][-1]*100))
        f.write('Best Validation: {0}\n'.format(np.max(hist.history['val_accuracy'])*100))
        f.write('\n')
        f.write('Modelarchitecture:\n')
        with redirect_stdout(f):
            model.summary()
    
    