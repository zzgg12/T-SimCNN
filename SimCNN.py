# coding:utf-8
# coding: utf-8
# --**Created by Cao on 2019**--
# *****Main Trainning*****
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import math
# import cv2 as cv
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        # self.val_loss = {'batch':[], 'epoch': []}
        # self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        # self.val_loss['batch'].append(logs.get('val_loss'))
        # self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        # self.val_loss['epoch'].append(logs.get('val_loss'))
        # self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='Train ACC')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='Train Loss')
        # if loss_type == 'epoch':
        #     # val_acc
        #     plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        #     # val_loss
        #     plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC-Loss')
        plt.legend(loc="upper right")
        plt.savefig("train_SimCNN.png")
        plt.show()

class Similar(object):

    def SimRank(self, Width, Height, k, img):

        img_open = Image.open('train_SimCNN_img_1/' + img)
        conv_RGB = img_open.convert('RGB')  # 统一转换一下RGB格式 统一化
        r, g, b = conv_RGB.split()
        r_array = np.array(r, np.uint8) / 255
        g_array = np.array(g, np.uint8) / 255
        b_array = np.array(b, np.uint8) / 255
        A_a_d_e = np.zeros((Width, Height, 3 * k))
        D_m_n = np.empty(Width)
        for m in range(0, Width):
            for n in range(0, Height):
                for i in range(0, Width):
                    if n == i:
                        sim = 1
                    else:
                        sim_d = 1 - math.sqrt(
                            ((r_array[m, n] - r_array[m, i]) ** 2 + (g_array[m, n] - g_array[m, i]) ** 2 + (
                                    b_array[m, n] - b_array[m, i]) ** 2) / 3)
                        sim = 0.5 / 5 * 5 * sim_d
                    D_m_n[i] = sim  
                Dmn = np.array(D_m_n)
                flat_indices = np.argpartition(Dmn.ravel(), k)[:k]  
               

                for aa in range(0, k):
                    # mm = row_indices[aa]
                    A_a_d_e[m, n, 3*aa] = r_array[m, flat_indices[aa]]
                    A_a_d_e[m, n, 3*aa + 1] = g_array[m, flat_indices[aa]]
                    A_a_d_e[m, n, 3*aa + 2] = b_array[m, flat_indices[aa]]
        print("完毕")
        return A_a_d_e


# main Training program
class Training(object):
    def __init__(self, batch_size, number_batch, categories, train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    def train(self):
        s = Similar()
        train_img_list = []  
        train_label_list = []  
        for file in os.listdir(self.train_folder):
            files_img_in_array = s.SimRank(Height=100, Width=100, k=5, img=file)
            train_img_list.append(files_img_in_array)  # Image list add up
            train_label_list.append(int(file.split('_')[0]))  # lable list addup

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)

        train_label_list = np_utils.to_categorical(train_label_list, self.categories)  

        train_img_list = train_img_list.astype('float32') 

        #  setup Neural network CNN
        model = Sequential()
        # CNN Layer - 1
        model.add(Convolution2D(
            input_shape=(100, 100, 15),
            filters=32, 
            kernel_size=(5, 5), 
            padding='same', 
        ))
        model.add(Activation('relu'))
       
        model.add(MaxPooling2D(
            pool_size=(2, 2),  # Output for next layer (50,50,32)
            strides=(2, 2),
            padding='same', 
        ))

        # CNN Layer - 2
        model.add(Convolution2D(
            filters=64,  # Output for next layer (50,50,64)
            kernel_size=(3, 3),
            padding='same',
        ))
        model.add(Activation('relu'))
        '''
        model.add(Convolution2D(
            filters=150,  # Output for next layer (50,50,64)
            kernel_size=(3, 3),
            padding='same',
        ))
        model.add(Activation('relu'))
        '''
        model.add(MaxPooling2D(  # Output for next layer (25,25,64)
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        ))

    # Fully connected Layer -1
        model.add(Flatten()) 
        model.add(Dense(1024))
        model.add(Activation('relu'))
    # Fully connected Layer -2
        model.add(Dense(512))
        model.add(Activation('relu'))
    # Fully connected Layer -3
        model.add(Dense(256))
        model.add(Activation('relu'))
    # Fully connected Layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))  # 分类
    # Define Optimizer
        adam = Adam(lr=0.0005)  # 学习率

    # Compile the model
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )

  
        history = LossHistory()


        model.fit(
            x=train_img_list,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            # validation_split=0.2,
            verbose=1,
            callbacks=[history],
        )
        # SAVE your work -model
        model.save('./EQSimCNN.h5')
        # 绘制acc-loss曲线
        history.loss_plot('epoch')


def MAIN():

    EQtype=['事件', '噪声']


    # Trainning Neural Network
    Train = Training(batch_size=200, number_batch=100, categories=2, train_folder='train_SimCNN_img_1/')
    Train.train()


if __name__ == "__main__":
    MAIN()






