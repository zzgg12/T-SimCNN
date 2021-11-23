# coding:utf-8
# coding: utf-8
# --**Created by Cao on 2019**--
# *****Main Trainning*****
import os
import keras
keras.losses
import numpy as np
import matplotlib.pyplot as plt
import warnings
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
from keras.layers import Flatten, LSTM
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
warnings.filterwarnings("ignore")



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
       

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
       

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
       

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='Train ACC')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='Train Loss')
  
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC-Loss')
        plt.legend(loc="upper right")
        plt.savefig("train_LSTM.png")
        plt.show()



class Training(object):
    def __init__(self, batch_size, number_batch, categories, train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    def train(self):
        train_LSTM_list = []  
        train_label_list = []  
        for subclass in os.listdir(self.train_folder):
            fullFilename = self.train_folder + subclass

            b = np.loadtxt(fullFilename, delimiter=" ", usecols=(5,), dtype=float)
            b = b[range(0, 50000, 50)]
      
            for m in range(0, 999):
                b[m, ] = b[m, ]/sum(b)
         
            train_LSTM_list.append(b) 
            train_label_list.append(int(subclass.split('_')[0]))  
            print("完毕")

        train_LSTM_list = np.array(train_LSTM_list)
        train_LSTM_list = train_LSTM_list.reshape(104, 1000, 1)  

        #train_img_list = np.concatenate(train_img_list, axis=0)
        #train_img_list = train_img_list.astype('float32')  

        train_label_list = np.array(train_label_list)
        train_label_list = np_utils.to_categorical(train_label_list,
                                                   self.categories) 

        model = Sequential()
        model.add(LSTM(500,
                       activation='relu',
                       input_shape=(1000, 1),
                       return_sequences=False
                       )
                  )
        #model.compile(optimizer='adam', loss="categorical_crossentropy")
        model.add(Dense(512))
        model.add(Activation('relu'))

        #model.fit(X, Y, epochs=2000, validation_split=0.2, batch_size=5)
    # Fully connected Layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))  
    # Define Optimizer
        adam = Adam(lr=0.00001) 

  
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )

 
        history = LossHistory()
        model.summary()
        SVG(model_to_dot(model).create(prog='dot', format='svg'))
        plot_model(model, to_file='LSTM_model.png', show_shapes=True, show_layer_names=True)

        # Fire up the network
        model.fit(
            x=train_LSTM_list,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[history],
        )
        # SAVE your work -model
        model.save('./EQLSTM.h5')
        

def MAIN():

    # Trainning Neural Network
    Train = Training(batch_size=11, number_batch=50, categories=2, train_folder='train_LSTM_data/')
    Train.train()

if __name__ == "__main__":
    MAIN()













