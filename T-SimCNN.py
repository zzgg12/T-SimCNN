# coding:utf-8
# coding: utf-8
# --**Created by Cao on 2019**--
# *****Main Trainning*****
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import math
import numpy as np
import tensorflow as tf
# import cv2 as cv
import tensorflow.python.keras.callbacks as tpkc
from PIL import Image
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.optimizers import Adam
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot, plot_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'



# 写一个LossHistory类，保存loss和acc
class LossHistory(tpkc.Callback):
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
        plt.savefig("train_T-SimCNN.png")
        plt.show()


class Similar(object):

    def SimRank(self, Width, Height, k, img):

        img_open = Image.open('train_T-SimCNN_img_1/'+img)
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
                    D_m_n[i] = sim  # 每个节点都找出相似距离的矩阵
                Dmn = np.array(D_m_n)
                flat_indices = np.argpartition(Dmn.ravel(), k)[:k]  # 要获取（展平的）二维数组中k个最小值的索引，
                # col_indices = np.unravel_index(flat_indices, Dmn.shape)

                for aa in range(0, k):
                    # mm = row_indices[aa]
                    A_a_d_e[m, n, 3 * aa] = r_array[m, flat_indices[aa]]
                    A_a_d_e[m, n, 3 * aa + 1] = g_array[m, flat_indices[aa]]
                    A_a_d_e[m, n, 3 * aa + 2] = b_array[m, flat_indices[aa]]
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
        train_img_list = []  # 图像列表
        train_label_list = []  # 对应标签列表
        for file in os.listdir(self.train_folder):
            files_img_in_array = s.SimRank(Height=100, Width=100, k=5, img=file)
            train_img_list.append(files_img_in_array)  # Image list add up
            train_label_list.append(int(file.split('_')[0]))  # lable list addup

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)

        train_label_list = np_utils.to_categorical(train_label_list, self.categories)  # format into binary[0,1]，[1,0]

        train_img_list = train_img_list.astype('float32')  # 转为浮点型

        ####导入模型
        best_model_1 = tf.keras.models.load_model('EQSimCNN.h5')
        best_model_2 = tf.keras.models.load_model('EQLSTM.h5')

        ####更改模型输出层，冻结其他层
        layer_1 = len(best_model_1.layers)
        layer_2 = len(best_model_2.layers)
        new_model = Sequential(best_model_1.layers[0:(layer_1 - 3)])
        new_model.add(Sequential(best_model_2.layers[layer_2 - 2:layer_2 - 2]))
        # Fully connected Layer -3
        new_model.add(Dense(256, name="dense_a"))
        new_model.add(Activation('relu'))
        # Fully connected Layer -4
        new_model.add(Dense(self.categories))
        new_model.add(Activation('softmax', name="dense_output"))  # 分类
        # Define Optimizer
        adam = Adam(lr=0.0005)  # 学习率

        ####控制哪些层重新训练
        for i in range(layer_1 - 3):
            new_model.layers[i].trainable = False
        for i in range(layer_2 - 2, layer_2 - 1):
            new_model.layers[i].trainable = False
        ####compile模型
        new_model.compile(optimizer=adam,
                          loss="categorical_crossentropy",
                          metrics=['accuracy']
                          )

        ####训练模型，输出相关信息


    # 创建一个实例
        history = LossHistory()
        new_model.summary()
        SVG(model_to_dot(new_model).create(prog='dot', format='svg'))
        plot_model(new_model, to_file='T-SimCNN_model.png', show_shapes=True, show_layer_names=True)

    # Fire up the network
        new_model.fit(
            x=train_img_list,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            # validation_split=0.2,
            verbose=1,
            callbacks=[history],
        )
        # SAVE your work -model
        new_model.save('./EQT-SimCNN.h5')
        # 绘制acc-loss曲线
        history.loss_plot('epoch')


def MAIN():

    EQtype=['事件', '噪声']


    # Trainning Neural Network
    Train = Training(batch_size=32, number_batch=100, categories=2, train_folder='train_T-SimCNN_img_1/')
    Train.train()


if __name__ == "__main__":
    MAIN()






