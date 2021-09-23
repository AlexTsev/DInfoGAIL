import tensorflow as tf
from myconfig import myconfig
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU
from tensorflow.python.keras.models import Model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

class Critic(object):

    def __init__(self, observation_dimensions=3):
        self.alpha = myconfig['critic_alpha']
        self.epochs = myconfig['critic_epochs']
        self.sess = tf.Session(config=config)
        self.inputs = Input(shape=(observation_dimensions,))

        h = Dense(100, activation='tanh')(self.inputs)
        h = Dense(100, activation='tanh')(h)

        self.out = Dense(1)(h)

        self.model = Model(inputs=self.inputs, outputs=self.out)
        self.predictions = self.model.outputs[0]
        self.labels = tf.placeholder(tf.float32, shape=(None), name='y')
        self.loss = tf.reduce_mean(tf.square(self.predictions - self.labels))
        self.opt = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y):
        for i in range(self.epochs):
            _, loss_run, _ = self.sess.run([self.opt, self.loss, self.labels], feed_dict={self.inputs: x, self.labels: y})
            # if i % 100 == 0: print(i, "loss:", loss_run)
            # print(i, "critic loss:", loss_run)

    def predict(self, x):
        return self.sess.run(self.model.outputs, feed_dict={self.inputs: x})[0]
