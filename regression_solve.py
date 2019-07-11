#!/usr/bin/python
#_*_coding:UTF-8_*_
'''
author:jiaxin jiang
time:23:16
date:1/19/2019
function:using validation data decide train model to test test data
'''
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
from sklearn.model_selection import StratifiedKFold
import string
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import multi_gpu_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import DataGenerator as dg
import get_modelv2_3
from keras.utils import plot_model
import sys
from keras import backend as K
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import scipy
import seaborn as sns
# sns.set(color_codes=True)
import re
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # n设置可见gpu7,3,4,0,1,4,23,4,5,66,7

class Regression_solve:
    def __init__(self):
        self.model=keras.Model()
        # legend字体
        self.font1 = {
                 'weight': 'normal',
                 'size': 16,
                 }
        # 横纵坐标标题字体
        self.font2 = {
                 'weight': 'normal',
                 'size': 23,
                 }
    def change_model_2_regression(self):
        '''
        特比的，当把分类模型最后输出的激活函数单独拿出来当做一层之后，将分类模型修改成回归模型，只需要把最后一层去掉就可以了
        '''
        layer_name = self.model.layers[-2].name#倒数第二层
        origin_outputs = self.model.get_layer(layer_name).output
        inputs = self.model.input
        # output = keras.layers.Dense(1, name='regression_output')(origin_outputs)  # activation='relu',
        self.model = keras.Model(inputs=inputs, outputs=origin_outputs)#outputs=output
        print("generate regression model succeed!")
        print(self.model.summary())

    def mean_squared_error_l2(self, y_true, y_pred, lmbda=0.01):
        cost = K.mean(K.square(y_pred - y_true))
        # weights = self.model.get_weights()
        weights = []
        for layer in self.model.layers:
            # print(layer)
            weights = weights + layer.get_weights()
            # print (weights)
        result = tf.reduce_sum([tf.reduce_sum(tf.pow(wi, 2)) for wi in weights])
        l2 = lmbda * result  # K.sum([K.square(wi) for wi in weights])
        return cost + l2

    def predict_result(self,x_prot,x_comp,y_value,type):
        predict_result = self.model.predict([x_prot, x_comp])
        real = y_value
        df = pd.DataFrame(predict_result, columns=['predicted'])
        df['real'] = real
        df['set'] = type
        return df

    def save_predict_result(self,x_prot,x_comp,y_value,model_name,type):
        # 保存预测结
        df=self.predict_result(x_prot,x_comp,y_value,type)
        if not os.path.exists('predict_value'):
            os.mkdir('predict_value')
        df.to_csv('predict_value/regression_model_%s_%s_predict_result.csv' % (model_name,type), index=False)

    def computer_parameter(self, df,type):
        # 计算参数，画散点图
        rmse = ((df['predicted'] - df['real']) ** 2).mean() ** 0.5
        mae = (np.abs(df['predicted'] - df['real'])).mean()
        corr = scipy.stats.pearsonr(df['predicted'], df['real'])
        lr = LinearRegression()
        lr.fit(df[['predicted']], df['real'])
        y_ = lr.predict(df[['predicted']])
        sd = (((df["real"] - y_) ** 2).sum() / (len(df) - 1)) ** 0.5
        print("%10s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f" % (type, rmse, mae, *corr, sd))
        return type, rmse, mae, corr, sd

    def computer_parameter_draw_scatter_plot(self,x_prot, x_comp, y_value,model_name,type):
        sns.set(context='paper', style='white')
        sns.set_color_codes()
        df = self.predict_result(x_prot, x_comp, y_value, type)
        if all(df['real']>0):
            xlimb_start=0
        else:
            xlimb_start=-10
        if all(df['predicted']>0):
            ylimb_start=0
        else:
            ylimb_start=-10
        set_colors = {'train': 'b', 'validation': 'green', 'test': 'purple'}
        grid = sns.jointplot('real', 'predicted', data=df, stat_func=None, color=set_colors[type],
                             space=0.0, size=4, ratio=4, s=20, edgecolor='w', ylim=(ylimb_start, 16),
                             xlim=(xlimb_start, 16)
                             )# s：点的大小;x-lim和y-lim
        grid.ax_joint.set_xticks(range(xlimb_start, 16, 5))
        grid.ax_joint.set_yticks(range(ylimb_start, 16, 5))  # 可单独画一张带负值的，但还是以真实正值为主
        type, rmse, mae, corr, sd=self.computer_parameter(df,type)
        grid.ax_joint.text(1, 14, type + ' set', fontsize=14)  # 调整标题大小
        grid.ax_joint.text(16, 19.5, 'RMSE: %.2f ' % rmse)
        grid.ax_joint.text(16, 18.5, '(p): %.3f ' % corr[1])
        grid.ax_joint.text(16, 17.5, 'R2: %.2f ' % corr[0])
        grid.ax_joint.text(16, 16.5, 'SD: %.2f ' % sd)
        grid.fig.savefig('%s_%s_scatter_plot.jpg' %(model_name,type), dpi=400)

    def draw_loss_change(self,history,model_name):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss_values, 'b', label='Training loss')
        plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
        plt.title('Training and validation loss', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig('%s_regression_training_validation_loss.png'%model_name)
        # plt.show()
        ##___________________________________________________________________-
        plt.figure(figsize=(10, 10))
        times = range(1, len(loss_values) + 1)
        plt.plot(times[10:(len(loss_values) + 1)], loss_values[10:(len(loss_values) + 1)], 'b',
                 label='Training loss')
        plt.plot(times[10:(len(loss_values) + 1)], val_loss_values[10:(len(loss_values) + 1)], 'r',
                 label='Validation loss')
        plt.tick_params(labelsize=20)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig('%s_regression_training_validation_loss2.png'%model_name)

    def read_data(self, data_type, file,reinforced=False):
        print("starting read %s data:" % data_type)
        x_prot, x_comp, y = dg.multi_process_read_pro_com_file_regression(file,reinforced=reinforced)
        print("%s data,%s, has been read succeed!" % (data_type, file))
        print('x_prot.shape', x_prot.shape)
        print('x_comp.shape', x_comp.shape)
        print('y.shape', y.shape)
        return x_prot, x_comp, y

    def train_model(self,train_file,validation_file,model_name,lr = 0.0001,batch_size = 512):
        # 一些设置项
        alpha=0.3
        epochs = 300  # 50
        patience = 10
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        print('1')
        # 读训练数据
        train_x_prot, train_x_comp, train_y=self.read_data("train",train_file,reinforced=False)#True
        #读验证集
        validation_x_prot, validation_x_comp, validation_y =self.read_data("validation",validation_file)
        # 生成模型回归模型
        self.model=get_modelv2_3.get_model9()#生成分类模型
        self.change_model_2_regression()# 修改最后一层，使之成为回归问题

        # regression model tain
        log_filepath = './tmp/keras_log/Xavier_uniform/'
        tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
        # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的
        # 权值，每层输出值的分布直方图
        optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(loss=self.mean_squared_error_l2, optimizer=optimizer,
                           metrics=['mse', 'mae'])#loss='mean_squared_error'
        ##早结束
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        ## checkpoint
        filepath = save_dir + "/loss-reduction-{epoch:02d}-{val_loss:.2f}-{val_mean_squared_error:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only='True',mode='min')
                                                                    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次,
        # 训练模型
        history = self.model.fit([train_x_prot, train_x_comp],
                            train_y,
                            shuffle=True,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=([validation_x_prot, validation_x_comp], validation_y),
                            callbacks=[early_stopping, checkpoint]  # 回调函数,,tb_cb
                            )
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name + '_regression_train.h5')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        # Score
        score = self.model.evaluate([validation_x_prot, validation_x_comp], validation_y)
        print(score)
        # saving predict value
        self.save_predict_result(train_x_prot, train_x_comp, train_y, model_name,'train')
        self.save_predict_result(validation_x_prot, validation_x_comp,validation_y, model_name,'validation')
        # train and validation loss change
        self.draw_loss_change(history,model_name)
        # computing parameters and drawing scatter plot
        self.computer_parameter_draw_scatter_plot(train_x_prot, train_x_comp, train_y, model_name,'train')
        self.computer_parameter_draw_scatter_plot(validation_x_prot, validation_x_comp, validation_y,model_name,
                                                  'validation')


    def load_model_test(self,model_file,test_file,train_file=None,validation_file=None):
        # read test data
        x_prot_test, x_comp_test, y_test =self.read_data("test",test_file)
        # load_model
        self.model = load_model(model_file, custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        # score on the data
        print("scoring on the test data:")
        result = self.model.evaluate([x_prot_test, x_comp_test], y_test)
        print("loss, mse and mae:", result)
        tmp= model_file.split('/')[-1]
        model_name=re.findall(r"(.+?).hdf5", tmp)[0]
        # saving predict value
        self.save_predict_result(x_prot_test, x_comp_test, y_test, model_name,'test')
        # computing parameters and drawing scatter plot
        self.computer_parameter_draw_scatter_plot(x_prot_test, x_comp_test, y_test, model_name,'test')
        #for train file
        if train_file != None:
            #read data
            x_prot_train, x_comp_train, y_train = self.read_data("train", train_file)
            #score
            print("scoring on the train data:")
            result = self.model.evaluate([x_prot_train, x_comp_train], y_train)
            print("loss, mse and mae:", result)
            # saving predict value
            self.save_predict_result(x_prot_train, x_comp_train, y_train, model_name, 'train')
            # computing parameters and drawing scatter plot
            self.computer_parameter_draw_scatter_plot(x_prot_train, x_comp_train, y_train, model_name, 'train')
        if validation_file != None:
            # read data
            x_prot_validation, x_comp_validation, y_validation = self.read_data("validation", validation_file)
            # score
            print("scoring on the validation data:")
            result = self.model.evaluate([x_prot_validation, x_comp_validation], y_validation)
            print("loss, mse and mae:", result)
            # saving predict value
            self.save_predict_result(x_prot_validation, x_comp_validation, y_validation, model_name, 'validation')
            # computing parameters and drawing scatter plot
            self.computer_parameter_draw_scatter_plot(x_prot_validation, x_comp_validation, y_validation,
                                                      model_name,'validation')

    def load_model_predict(self,model_file,file):
        # read data
        x_prot, x_comp, y_label = dg.multi_process_read_pro_com_file(file)
        # load_model
        self.model = load_model(model_file, custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        tmp= model_file.split('/')[-1]
        model_name=re.findall(r"(.+?).hdf5", tmp)[0]
        # saving predict value
        self.save_predict_result(x_prot, x_comp, y_label, model_name,'none')

def main():
    if len(sys.argv)==4:
        #model train
        train_file = sys.argv[1]
        validation_file = sys.argv[2]
        model_name = sys.argv[3]
        print("train data is", train_file)
        regression_model=Regression_solve()
        regression_model.train_model(train_file,validation_file,model_name)
    elif len(sys.argv)==3:
        # model test
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        print("test data is", test_file)
        regression_model = Regression_solve()
        regression_model.load_model_test(model_file, test_file)
        # regression_model.load_model_predict(model_file, test_file)
    elif len(sys.argv)==5:
        # model test
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        train_file = sys.argv[3]
        validation_file = sys.argv[4]
        regression_model = Regression_solve()
        regression_model.load_model_test(model_file, test_file,train_file,validation_file)
    else:
        #
        print("input parametes not illegal, please chec   k and reinput!")
        train_file ='../dataset/dataset_reg_train.txt'
        validation_file = '../dataset/dataset_reg_vali.txt'
        model_name = 'mode9_on_dataset_reg_4.25'
        print("train data is", train_file)
        regression_model = Regression_solve()
        regression_model.train_model(train_file, validation_file, model_name)


if __name__ == '__main__':
    main()