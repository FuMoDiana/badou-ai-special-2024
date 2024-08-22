from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
# 兼容tf版本
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.latent_dim = 100 # 生成器输入的噪声向量的维度
        # 优化器
        optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        # 生成器
        self.generator = self.build_generator()
        # 判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # 组合模型
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)

        # 堆叠生成器和判别器,用于训练生成器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 构建判别器模型
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary() # 打印模型结构

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img,validity)
    # 构建生成器模型
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise,img)
    def train(self,epochs, batch_size=128, sample_interval=50):
        # 加载数据集数据
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train / 127.5 -1 # 归一化[-1,1]
        X_train = np.expand_dims(X_train, axis=3)

        # 真假标签对应本批次的每一张图
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            '''
                真实图片和生成图片训练判别器
            '''
            # 选择一批真实图像
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 生成一批新的假图像
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # 真实图片和虚假图片分别训练判别器
            # 1.学习真实图片 valid=1
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # 2. 识别虚假图片 fake=0
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # 上面两步的平均损失
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            '''
                通过训练生成器来欺骗判别器，使判别器将生成器生成的假图像判定为真实图像
                组合模型中，固定了判别器，因此判别器的参数不会更新  self.discriminator.trainable = False
            '''
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # 更新生成器的参数
            g_loss = self.combined.train_on_batch(noise, valid)

            print('%d [D loss: %f, acc: %.2f] [G loss: %f]' %(epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # 保存生成的图片样本,每50个批次保存一次
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        # 生成图片的行数和列数
        r,c = 5,5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        #输入噪声，生成器生成图片
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 # 像素值缩放到[0,1]
        fig, axs = plt.subplots(r, c) # 生成r*c形状的网格展示生成的图片
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
