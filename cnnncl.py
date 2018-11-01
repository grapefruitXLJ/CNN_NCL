# -*- coding:utf-8 -*-
import tensorflow as tf
import scipy.io as sio
import random
import math
import matplotlib.pyplot as plt
import os
import numpy as np


class CornSample:
    def __init__(self, gpu=1, batch_size=16, lr=0.001):
        self.batch_size = batch_size
        self.checkpoint_dir = "./model0_0"
        self.model_name = "cnnncl0.0"
        self.lr = lr  # learning rate
        self.a = 0.0  # penalty coefficient
        # if gpu:
        #     config = tf.ConfigProto()
        #     config.gpu_options.per_process_gpu_memory_fraction = 0.5
        #     self.sess = tf.Session(config=config)
        # else:
        #     self.sess = tf.Session()
        self.sess = tf.Session()
        self.build()

    def conv1d(self, x, out, k=5, stride=1):
        w = tf.get_variable("conv2d", initializer=tf.random_normal(shape=[k, x.get_shape()[-1].value, out], stddev=0.1))
        b = tf.get_variable("bias", [out], initializer=tf.constant_initializer(0))
        conv = tf.nn.conv1d(x, w, stride=stride, padding='VALID')
        return tf.nn.bias_add(conv, b)

    def relu(self, x):
        return tf.nn.relu(x)

    def max_pool(self, x, k=2, stride=2):
        return tf.nn.pool(x, window_shape=[k], pooling_type="MAX", strides=[stride], padding="VALID")

    def dense(self, x, out):
        x_ = tf.reshape(x, [x.get_shape()[0].value, -1])
        w = tf.get_variable("dense", initializer=tf.random_normal(shape=[x_.get_shape()[-1].value, out], stddev=0.1))
        b = tf.get_variable("bias", [out], initializer=tf.constant_initializer(0))
        return tf.matmul(x_, w)+b

    def inference_1(self, x):
        with tf.variable_scope("cnn1_conv1"):
            net1 = self.relu(self.conv1d(x, 64))
        with tf.name_scope("cnn1_pool1"):
            net2 = self.max_pool(net1, 4, 4)
        with tf.variable_scope("cnn1_conv2"):
            net3 = self.relu(self.conv1d(net2, 128))
        with tf.name_scope("cnn1_pool2"):
            net4 = self.max_pool(net3, 4, 4)
        with tf.variable_scope("cnn1_conv3"):
            net5 = self.relu(self.conv1d(net4, 256))
        with tf.name_scope("cnn1_pool3"):
            net6 = self.max_pool(net5, 4, 4)
        with tf.variable_scope("cn"
                               "n1_Dense"):
            net7 = self.dense(net6, 1)
        return net7

    def inference_2(self, x):
        with tf.variable_scope("cnn2_conv1"):
            net1 = self.relu(self.conv1d(x, 64))
        with tf.name_scope("cnn2_pool1"):
            net2 = self.max_pool(net1, 4, 4)
        with tf.variable_scope("cnn2_conv2"):
            net3 = self.relu(self.conv1d(net2, 128))
        with tf.name_scope("cnn2_pool2"):
            net4 = self.max_pool(net3, 4, 4)
        with tf.variable_scope("cnn2_conv3"):
            net5 = self.relu(self.conv1d(net4, 256))
        with tf.name_scope("cnn2_pool3"):
            net6 = self.max_pool(net5, 4, 4)
        with tf.variable_scope("cnn2_Dense"):
            net7 = self.dense(net6, 1)
        return net7

    def inference_3(self, x):
        with tf.variable_scope("cnn3_conv1"):
            net1 = self.relu(self.conv1d(x, 64))
        with tf.name_scope("cnn3_pool1"):
            net2 = self.max_pool(net1, 4, 4)
        with tf.variable_scope("cnn3_conv2"):
            net3 = self.relu(self.conv1d(net2, 128))
        with tf.name_scope("cnn3_pool2"):
            net4 = self.max_pool(net3, 4, 4)
        with tf.variable_scope("cnn3_conv3"):
            net5 = self.relu(self.conv1d(net4, 256))
        with tf.name_scope("cnn3_pool3"):
            net6 = self.max_pool(net5, 4, 4)
        with tf.variable_scope("cnn3_Dense"):
            net7 = self.dense(net6, 1)
        return net7

    def build(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 700, 1], name='Inputdata')
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 1], name='Trueout')
        with tf.variable_scope("cnn1"):
            self.out_1 = self.inference_1(self.x)
        with tf.variable_scope("cnn2"):
            self.out_2 = self.inference_2(self.x)
        with tf.variable_scope("cnn3"):
            self.out_3 = self.inference_3(self.x)
        var1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn1")
        var2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn2")
        var3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn3")

        self.out = (self.out_1 + self.out_2 + self.out_3)/3
        self.loss_ = tf.reduce_sum(tf.square(self.out - self.y))
        self.loss_1 = tf.reduce_mean(tf.square(self.out_1 - self.y)*0.5) \
                      + self.a * 0.5*tf.reduce_mean((self.out_1-self.out)*(self.out_2 + self.out_3 - 2*self.out))

        self.loss_2 = tf.reduce_mean(tf.square(self.out_2 - self.y)*0.5) \
                      + self.a * 0.5*tf.reduce_mean((self.out_2-self.out)*(self.out_1 + self.out_3 - 2*self.out))

        self.loss_3 = tf.reduce_mean(tf.square(self.out_3 - self.y)*0.5) \
                      + self.a * 0.5*tf.reduce_mean((self.out_3-self.out)*(self.out_1+self.out_2 - 2*self.out))
        self.optm_1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_1, var_list=var1)
        self.optm_2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_2, var_list=var2)
        self.optm_3 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_3, var_list=var3)

        self.saver = tf.train.Saver(max_to_keep=2)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, nEpochs, load_model=False):
        if load_model:
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        # load data
        datamat = sio.loadmat("E:\\xlj\\10.17data\\N0.mat")
        X_train = datamat['Csample']
        Y_train = datamat['Cpropval'][:, 0]
        X_test = datamat['Vsample']
        Y_test = datamat['Vpropval'][:, 0]
        # start training
        Rmsec = np.array(None, dtype=float)
        Rmsecv = np.array(None, dtype=float)
        for epoch in range(nEpochs):
            li = [i for i in range(X_train.shape[0])]
            lt = [i for i in range(X_test.shape[0])]
            random.shuffle(li)
            random.shuffle(lt)
            data_ = X_train[li, :]
            label_ = Y_train[li]
            data_test = X_test[lt, :]
            label_test = Y_test[lt]
            RMSEC = 0
            RMSECV = 0
            N = math.floor(X_train.shape[0] / self.batch_size)
            for i in range(N):
                x_train = data_[i*self.batch_size:(i+1)*self.batch_size, :]
                x_train = x_train[..., None]
                l_trian = label_[i*self.batch_size:(i+1)*self.batch_size]
                l_trian = l_trian[..., None]
                _, __, ___, loss_ = self.sess.run([self.optm_1, self.optm_2, self.optm_3, self.loss_], feed_dict={self.x: x_train, self.y: l_trian})
                RMSEC = RMSEC + loss_
            Rmsec = np.append(Rmsec, np.sqrt(RMSEC / (N * self.batch_size)))
            print("nEpochs/epoch:%d/%d RMSEC:%f" % (nEpochs, epoch, np.sqrt(RMSEC / (N * self.batch_size))))
            # validation every 5 epoch
            if (epoch+1) % 5 == 0:
                M = math.floor(X_test.shape[0] / self.batch_size)
                for j in range(M):
                    x_test = data_test[j * self.batch_size:(j + 1) * self.batch_size, :]
                    x_test = x_test[..., None]
                    y_test = label_test[j * self.batch_size:(j + 1) * self.batch_size]
                    y_test = y_test[..., None]
                    loss_, y_pre = self.sess.run([self.loss_, self.out], feed_dict={self.x: x_test, self.y: y_test})
                    RMSECV = RMSECV + loss_
                Rmsecv = np.append(Rmsecv, np.sqrt(RMSECV / (M * self.batch_size)))
                print("RMSECV:%f" % (np.sqrt(RMSECV / (M * self.batch_size))))
        # plot rmsec and rmsecv to detect if the model is over-fitted
        plt.subplot(121)
        plt.axis([0, 2000, 0, 1])
        plt.xlabel('Epoch')
        plt.ylabel('RMSEC')
        plt.plot(Rmsec)
        plt.subplot(122)
        plt.axis([0, 400, 0, 1])
        plt.xlabel('Epoch')
        plt.ylabel('RMSECV')
        plt.plot(Rmsecv)
        plt.show()
        # save model
        self.save(self.checkpoint_dir, epoch)
        # save Rmsec,Rmsec
        sio.savemat("E:\\xlj\\10.17data\\out0\\a0cv.mat", {'rmsec': Rmsec, 'rmsecv': Rmsecv})

    def test(self, epochs=50):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        datamat = sio.loadmat("E:\\xlj\\10.17data\\N0.mat")
        X_test = datamat['Psample']
        Y_test = datamat['Ppropval'][:, 0]
        Rmsep = np.array(None, dtype=float)
        for epoch in range(epochs):
            lt = [i for i in range(X_test.shape[0])]
            random.shuffle(lt)
            data_test = X_test[lt, :]
            label_test = Y_test[lt]
            RMSEP = 0
            M = math.floor(X_test.shape[0] / self.batch_size)
            for j in range(M):
                x_test = data_test[j * self.batch_size:(j + 1) * self.batch_size, :]
                x_test = x_test[..., None]
                y_test = label_test[j * self.batch_size:(j + 1) * self.batch_size]
                y_test = y_test[..., None]
                loss_, y_pre = self.sess.run([self.loss_, self.out], feed_dict={self.x: x_test, self.y: y_test})
                RMSEP = RMSEP + loss_
            Rmsep = np.append(Rmsep, np.sqrt(RMSEP / (M * self.batch_size)))
            print("RMSEP:%f" % (np.sqrt(RMSEP / (M * self.batch_size))))

        sio.savemat("E:\\xlj\\10.17data\\out0\\a0p.mat", {'rmsep': Rmsep})


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))

            return True
        else:
            return False


if __name__ == "__main__":
    model = CornSample(lr=0.001, batch_size=8)
    model.train(2000, load_model=True)

    model = CornSample(batch_size=3)
    model.test()