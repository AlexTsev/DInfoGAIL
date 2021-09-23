#from __future__ import print_function
#matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, LeakyReLU, InputLayer, Softmax
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from environment import environment
#import tensorflow.contrib.slim as slim #works with 1.14 tensorflow
#import tensorflow_probability as tf_prob
#import tensorflow_datasets as tfds
from sklearn import preprocessing
from sklearn.utils import shuffle as shuffle1
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import os
import time
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
from matplotlib import pyplot as plt
import numpy as np
from myconfig import myconfig
from utils import conjugate_gradient, set_from_flat, kl, self_kl, \
    flat_gradient, get_flat, discount, line_search, gauss_log_prob, visualize, gradient_summary, \
    unnormalize_action, unnormalize_observation

#sns.set_style('whitegrid')
sns.set()
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'Blues'

# Define the different distributions
distributions = tf.contrib.distributions
bernoulli = distributions.Bernoulli
#distributions = tf_prob.distributions
#bernoulli = distributions.Bernoulli

# Define current_time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
# GPU Usage

config = tf.ConfigProto()#(allow_soft_placement=False)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.7


# Define Directory Parameters
flags = tf.app.flags
flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/', 'Directory for data')
flags.DEFINE_string('results_dir', os.getcwd() + './results_gumbel_softmax/', 'Directory for results')
flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/results_gumbel_softmax/checkpoint/' + 'run2(batch-32-modes3)', 'Directory for checkpoints')#current_time
flags.DEFINE_string('models_dir', os.getcwd() + '/results_gumbel_softmax/model/', 'Directory for models')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
flags.DEFINE_integer('num_iters', 50000, 'Number of iterations')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs')
#flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')#0.0001
#flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
#flags.DEFINE_integer('num_cat_dists', 200, 'Number of categorical distributions') # num_cat_dists//num_calsses
flags.DEFINE_integer('num_actions', 3, 'Number of classes')
flags.DEFINE_float('init_temp', 5.0, 'Initial temperature')
flags.DEFINE_float('min_temp', 0.1, 'Minimum temperature')#0.5
#flags.DEFINE_float('init_temp', 1.0, 'Initial temperature')
#flags.DEFINE_float('min_temp', 0.5, 'Minimum temperature')
#flags.DEFINE_float('anneal_rate', 0.003, 'Anneal rate')
flags.DEFINE_float('anneal_rate', 0.00003, 'Anneal rate')
#flags.DEFINE_float('anneal_rate', 5e-4, 'Anneal rate')
#flags.DEFINE_float('anneal_rate', 3e-4, 'Anneal rate')
flags.DEFINE_bool('straight_through', False, 'Straight-through Gumbel-Softmax')
flags.DEFINE_string('kl_type', 'relaxed', 'Kullback-Leibler divergence (relaxed or categorical)')
flags.DEFINE_bool('learn_temp', False, 'Learn temperature parameter')

flags.DEFINE_integer('encoder_n_input', 10, 'input size')
flags.DEFINE_integer('n_hidden_encoder_layer_1', 100, 'encoder_layer1 size')
flags.DEFINE_integer('n_hidden_encoder_layer_2', 100, 'encoder_layer2 size')
flags.DEFINE_integer('latent_nz', 3, 'latent size')
#flags.DEFINE_integer('latent_nz', 5, 'latent size')
flags.DEFINE_integer('decoder_n_input', 10, 'input size')
flags.DEFINE_integer('n_hidden_decoder_layer_1', 100, 'decoder_layer1 size')
flags.DEFINE_integer('n_hidden_decoder_layer_2', 100, 'decoder_layer2 size')
FLAGS = flags.FLAGS


# WEIGHTS INITIALIZATION
def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float64)

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# Normalize
def apply_normalization_observations(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm_obs = pd.DataFrame(np_scaled, columns=['longitude', 'latitude', 'altitude', 'timestamp', 'Pressure_surface', 'Relative_humidity_isobaric', 'Temperature_isobaric', 'Wind_speed_gust_surface', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric'])
    return df_norm_obs

def apply_normalization_actions(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm = pd.DataFrame(np_scaled, columns=['dlon', 'dlat', 'dalt'])
    return df_norm

#OneHotEncoder
def apply_oneHotEncoder(X,y):
    print('OneHotEncoding...')
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    #columnTransformer = ColumnTransformer([('encoder', preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
    #data_encoded = np.array(columnTransformer.fit_transform((df), dtype=np.str))
    data_encoded = one_hot_encoder.fit_transform(X,y)
    data_encoded = np.array(data_encoded)
    return data_encoded

def pandas1_enriched(obs):
    df_norm_obs = pd.DataFrame(obs, columns=['longitude', 'latitude', 'altitude', 'timestamp', 'Pressure_surface', 'Relative_humidity_isobaric', 'Temperature_isobaric', 'Wind_speed_gust_surface', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'bearing', 'constant_speed', 'great_circle_distance', 'temporal_distance', 'v_speed', 'isobaric_level', 'Cluster', 'drct', 'sknt', 'alti', 'vsby', 'gust'])
    return df_norm_obs

def pandas1(obs):
    df_norm_obs = pd.DataFrame(obs, columns=['longitude', 'latitude', 'altitude', 'timestamp', 'Pressure_surface', 'Relative_humidity_isobaric', 'Temperature_isobaric', 'Wind_speed_gust_surface', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric'])
    return df_norm_obs

def pandas2(actions):
    df_norm = pd.DataFrame(actions, columns=['dlon', 'dlat', 'dalt'])
    return df_norm

def normalize_observations(obs):
    obs['longitude'] = (obs['longitude']-myconfig['longitude_avg'])/myconfig['longitude_std']
    obs['latitude'] = (obs['latitude'] - myconfig['latitude_avg']) / myconfig['latitude_std']
    obs['altitude'] = (obs['altitude'] - myconfig['altitude_avg']) / myconfig['altitude_std']
    obs['timestamp'] = (obs['timestamp'] - myconfig['timestamp_avg']) / myconfig['timestamp_std']

    obs['Pressure_surface'] = (obs['Pressure_surface'] - myconfig['Pressure_surface_avg']) / myconfig['Pressure_surface_std']
    obs['Relative_humidity_isobaric'] = (obs['Relative_humidity_isobaric'] - myconfig['Relative_humidity_isobaric_avg']) / myconfig['Relative_humidity_isobaric_std']
    obs['Temperature_isobaric'] = (obs['Temperature_isobaric'] - myconfig['Temperature_isobaric_avg']) / myconfig['Temperature_isobaric_std']
    obs['Wind_speed_gust_surface'] = (obs['Wind_speed_gust_surface'] - myconfig['Wind_speed_gust_surface_avg']) / myconfig['Wind_speed_gust_surface_std']
    obs['u-component_of_wind_isobaric'] = (obs['u-component_of_wind_isobaric'] - myconfig['u-component_of_wind_isobaric_avg']) / myconfig['u-component_of_wind_isobaric_std']
    obs['v-component_of_wind_isobaric'] = (obs['v-component_of_wind_isobaric'] - myconfig['v-component_of_wind_isobaric_avg']) / myconfig['v-component_of_wind_isobaric_std']

    return obs

def normalize_actions(actions):
    actions['dlon'] = (actions['dlon']-myconfig['dlon_avg'])/myconfig['dlon_std']
    actions["dlat"] = (actions['dlat'] -
                                       myconfig['dlat_avg']) / \
                                       myconfig['dlat_std']
    actions["dalt"] = (actions['dalt'] - myconfig['dalt_avg']) / \
                                    myconfig['dalt_std']

    return actions

def starting_points_2(observations,fname):
    observations_prcnt = np.empty((4, 0)).tolist()
    trajectories_prcnt = np.empty((4, 0)).tolist()
    n = [0,0.2,0.5,0.7]
    for name, group in observations.groupby('trajectory_ID'):
        # n = int(group.shape[0] / 5)

        for i in range(4):
            observations_prcnt[i].append(group.iloc[int(n[i]*group.shape[0]), :].values.tolist())
            trajectories_prcnt[i].extend(group.iloc[int(n[i]*group.shape[0]):, :].values.tolist())

    for i in range(4):
        pd.DataFrame(observations_prcnt[i], columns=['trajectory_ID', 'longitude', 'latitude',
                                                     'altitude', 'timestamp', 'Pressure_surface',
                                                     'Relative_humidity_isobaric',
                                                     'Temperature_isobaric',
                                                     'Wind_speed_gust_surface',
                                                     'u-component_of_wind_isobaric',
                                                     'v-component_of_wind_isobaric']).to_csv(
            myconfig['input_dir'] + '/' + str(int(n[i]*100)) +
            '%_' + fname + '_starting_points.csv',
            index=False,
        )
        pd.DataFrame(trajectories_prcnt[i], columns=['trajectory_ID', 'longitude', 'latitude',
                                                     'altitude', 'timestamp', 'Pressure_surface',
                                                     'Relative_humidity_isobaric',
                                                     'Temperature_isobaric',
                                                     'Wind_speed_gust_surface',
                                                     'u-component_of_wind_isobaric',
                                                     'v-component_of_wind_isobaric']).to_csv(
            myconfig['input_dir'] + '/' + str(int(n[i]*100)) +
            '%_' + fname + '_trajectories.csv',
            index=False,
        )
# Generate data from latent space
def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float64)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    #y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y = tf.keras.activations.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y



def encoder(x, latent, temperature):
    # Variational posterior q(y|x), i.e. the encoder (shape=(batch_size, 80))
    i=0

    h1 = Dense(FLAGS.n_hidden_encoder_layer_1, activation='tanh', kernel_initializer='glorot_uniform',bias_initializer='zeros', name='encoder_h1')
    h2 = Dense(FLAGS.n_hidden_encoder_layer_2, activation='tanh', kernel_initializer='glorot_uniform',bias_initializer='zeros', name='encoder_h2')
    out = Dense(FLAGS.latent_nz, name='encoder_out')
    #enc = tf.zeros([1,14], tf.float64)
    #enc = tf.Variable([[1.21717712e+00, -1.07320348e-04, -4.18256945e-03, -3.89150316e-03,
    #   -2.80910121e-03,  3.92307339e-03,  2.90272282e-03, -4.71132621e-03,
    #   -3.49365975e-04,  5.18269952e-05,  4.63955574e-03,  1.,
    #    0.,  0.]], dtype=tf.float64)

    q_y_list=[]
    log_q_y_list = []
    out_layer_list = []
    enc_inplayer_list = []
    while(i<32):
        #print('i:', i)
        feature = tf.gather(x, [i])
        enc_inp = tf.concat([feature, latent], 1)
        e = enc_inp
        #enc = tf.concat([enc, enc_inp], 0)
        # TENSORFLOW Keras Layers
        encoder_input_layer = Input(shape=(13,), batch_size=1, tensor=enc_inp, dtype=tf.float64)
        #encoder_input_layer = Input(shape=(15,), batch_size=1, tensor=enc_inp, dtype=tf.float64)
        enc_inplayer_list.append(encoder_input_layer)
        _h1 = h1(encoder_input_layer)
        _h2 = h2(_h1)
        _out = out(_h2)#, activation='softmax'
        out_layer_list.append(_out)
        #print('OUT:', out)
        #q_y = Softmax(out)
        q_y = tf.keras.activations.softmax(_out)
        q_y_temp = q_y
        q_y_list.append(q_y)

        log_q_y = tf.log(q_y + 1e-20)
        log_q_y_temp = log_q_y
        log_q_y_list.append(log_q_y)
        z = tf.reshape(gumbel_softmax(_out, temperature, hard=True), [-1, FLAGS.latent_nz])
        latent = z
        if(i<1):
            #q_y_list = tf.concat([q_y_temp, q_y], axis=0)
            #log_q_y_list = tf.concat([log_q_y_temp, log_q_y], axis=0)

            latents = tf.concat([latent, z], axis=0)#tf.Variable(z)
            enc = tf.concat([e, enc_inp], 0)
        else:
            #q_y_list = tf.concat([q_y_list, q_y], axis=0)
            #log_q_y_list = tf.concat([log_q_y_list, log_q_y], axis=0)

            latents = tf.concat([latents, z], axis=0)
            enc = tf.concat([enc, enc_inp], 0)


        i+=1
    # Encoder Model
    encoder_model = Model(inputs=enc_inplayer_list, outputs=out_layer_list)

    enc = tf.unstack(enc, 33)
    latents = tf.unstack(latents, 33)
    #q_y_list = tf.unstack(q_y_list, 33)
    #log_q_y_list = tf.unstack(log_q_y_list, 33)
    del enc[0]
    del latents[0]
    #del q_y_list[0]
    #del log_q_y_list[0]
    latents = latents
    enc = enc
    #q_y_list = q_y_list
    #log_q_y_list = log_q_y_list
    latents_shape = tf.shape(latents)
    enc_shape = tf.shape(enc)

    print('q_y_list:', q_y_list)
    print('log_q_y_list:', log_q_y_list)
    print('len1:', len(q_y_list))
    print('len2:', len(log_q_y_list))


    return out_layer_list, q_y_list, log_q_y_list, encoder_model, enc, latents, latents_shape, enc_shape

def decoder(x, latents):
    # Generative model p(x|y), i.e. the decoder (shape=(batch_size, 200))

    #y = tf.reshape(gumbel_softmax(logits_y, temperature, hard=False), [-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    #z = tf.reshape(gumbel_softmax(out, temperature, hard=True), [-1, FLAGS.latent_nz])

    dec_inp = tf.concat([x, latents], 1)

    #Tensorflow Keras Layers
    decoder_input_layer = Input(shape=(13,), batch_size=32, tensor=dec_inp, dtype=tf.float64)
    dec_h1 = Dense(FLAGS.n_hidden_decoder_layer_1, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decoder_h1')(decoder_input_layer) #activation=tf.keras.layers.LeakyReLU(alpha=0.01), glorot_normal
    dec_h2 = Dense(FLAGS.n_hidden_decoder_layer_2, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decoder_h2')(dec_h1)
    #decoder_out = Dense(FLAGS.num_actions, name='decoder_out')(dec_h2)
    decoder_out = Dense(FLAGS.num_actions, name='decoder_out')(dec_h2)
    #decoder_model = Model(inputs=logits_y, outputs=decoder_out)
    softmax = tf.keras.activations.softmax(decoder_out)

    #decoder_out = tf.layers.flatten(decoder_out)
    #p_x = bernoulli(logits=softmax)#logits_x

    # Decoder Model
    decoder_model = Model(inputs=decoder_input_layer, outputs=decoder_out)

    return decoder_out, softmax, decoder_model, dec_inp


def create_train_op(y, lr, predicted):
    #NA TO DW
    loss = tf.reduce_mean(tf.square(predicted - y))
    #elbo = tf.reduce_sum(tf.square(predicted - y))
    #loss = tf.reduce_mean(elbo)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return train_op, loss

def create_train_op_kl(y, predicted, lr, q_y, log_q_y):
    #kl_tmp = tf.reshape(q_y * (log_q_y - tf.log(1.0 / FLAGS.num_classes)),[-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    #KL = tf.reduce_sum(kl_tmp, [1,2])
    #WITH KL DIVERGENCE
    #elbo = tf.reduce_sum(p_x.log_prob(x), 1) - KL
    #loss = tf.reduce_mean(-elbo)
    #(tf.log(1.0 / 5)
    kl_tmp = tf.reshape(q_y * (log_q_y - (tf.cast(tf.log(1.0/5), tf.float64) ) ), [-1,5])
    KL = tf.reduce_sum(kl_tmp, 1)
    #KL = tf.reduce_sum(kl_tmp, 1) @@MAUTO KANEI TRAIN
    elbo = tf.reduce_sum(tf.square(predicted - y)) - KL
    loss = tf.reduce_mean(-elbo)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return train_op, loss, (tf.cast(tf.log(1.0), tf.float64)), kl_tmp, KL, elbo

class Dataset:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def count_train_num_examples(self):
        return len(self.x_train), len(self.y_train)

    def count_test_num_examples(self):
        return len(self.x_test), len(self.y_test)

    def train_next_batch(self, counter):
        batch = self.x_train[counter:counter+32]
        batch_y = self.y_train[counter:counter+32]
        return batch, batch_y

    def test_next_batch(self, counter):
        batch = self.x_test[counter:counter+32]
        batch_y = self.y_test[counter:counter+32]
        return batch, batch_y

##########################################################################################

#PREPROCESSING

print('Reading Datasets...')
df_trajectories = pd.read_csv('../dataset/train_set.csv')#, index_col='trajectory_ID')
df_trajectories_noID = df_trajectories.drop(columns=['trajectory_ID'], axis=1)
train_starting_points = pd.read_csv('../dataset/raw/train_starting_points.csv')
test_trajectories = pd.read_csv('../dataset/test_set.csv')

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 477)

print(df_trajectories.head())
print(df_trajectories.columns)
print('shape', df_trajectories.shape)

features = ['longitude', 'latitude', 'altitude', 'timestamp', 'Pressure_surface', 'Relative_humidity_isobaric', 'Temperature_isobaric', 'Wind_speed_gust_surface', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']
features2 = ['bearing', 'constant_speed', 'great_circle_distance', 'temporal_distance', 'v_speed', 'isobaric_level', 'Cluster', 'drct', 'sknt', 'alti', 'vsby', 'gust']#, 'isobaric_level', 'Cluster', 'drct'  CLUSTER REMOVED
actions_features = ['dlon', 'dlat', 'dalt']

print("train.shape:", df_trajectories.shape)

X = df_trajectories_noID[features]
X2 = df_trajectories_noID[features2]
actions = df_trajectories_noID[actions_features]

#print('y:', y.head)
#print('Dataset without TrajectoryID:')
#print(X.head)

###Preprocess file#######################################

obs_train_df = df_trajectories[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                     'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
                                     'u-component_of_wind_isobaric','v-component_of_wind_isobaric']]

obs_test_df = test_trajectories[['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp', 'Pressure_surface',
                                       'Relative_humidity_isobaric', 'Temperature_isobaric', 'Wind_speed_gust_surface',
                                       'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']]

starting_points_2(obs_test_df, 'test')
test_flight_num = obs_test_df['trajectory_ID'].nunique()
obs_test = obs_test_df.drop(['trajectory_ID'], axis=1)
actions_test = test_trajectories[['dlon','dlat','dalt']]
print('test')
print(obs_test.head())
print(actions_test.head())

obs_train_df.groupby('trajectory_ID').head(1).to_csv('../dataset/train_starting_points.csv', index=False)

obs_train = obs_train_df.drop(['trajectory_ID'], axis=1)
actions_train = df_trajectories[['dlon','dlat','dalt']]

print(obs_train.head())
print(actions_train.head())

myconfig['dlon_avg'] = actions_train.loc[:, "dlon"].mean()
myconfig['dlon_std'] = actions_train.loc[:, "dlon"].std()
myconfig['dlat_avg'] = actions_train.loc[:, "dlat"].mean()
myconfig['dlat_std'] = actions_train.loc[:, "dlat"].std()
myconfig['dalt_avg'] = actions_train.loc[:, "dalt"].mean()
myconfig['dalt_std'] = actions_train.loc[:, "dalt"].std()

myconfig['longitude_avg'] = obs_train.loc[:, "longitude"].mean()
myconfig['longitude_std'] = obs_train.loc[:, "longitude"].std()
myconfig['latitude_avg'] = obs_train.loc[:, "latitude"].mean()
myconfig['latitude_std'] = obs_train.loc[:, "latitude"].std()
myconfig['altitude_avg'] = obs_train.loc[:, "altitude"].mean()
myconfig['altitude_std'] = obs_train.loc[:, "altitude"].std()
myconfig['timestamp_avg'] = obs_train.loc[:, "timestamp"].mean()
myconfig['timestamp_std'] = obs_train.loc[:, "timestamp"].std()

myconfig['Pressure_surface_avg'] = obs_train.loc[:, "Pressure_surface"].mean()
myconfig['Pressure_surface_std'] = obs_train.loc[:, "Pressure_surface"].std()
myconfig['Relative_humidity_isobaric_avg'] = obs_train.loc[:, "Relative_humidity_isobaric"].mean()
myconfig['Relative_humidity_isobaric_std'] = obs_train.loc[:, "Relative_humidity_isobaric"].std()
myconfig['Temperature_isobaric_avg'] = obs_train.loc[:, "Temperature_isobaric"].mean()
myconfig['Temperature_isobaric_std'] = obs_train.loc[:, "Temperature_isobaric"].std()
myconfig['Wind_speed_gust_surface_avg'] = obs_train.loc[:, "Wind_speed_gust_surface"].mean()
myconfig['Wind_speed_gust_surface_std'] = obs_train.loc[:, "Wind_speed_gust_surface"].std()
myconfig['u-component_of_wind_isobaric_avg'] = obs_train.loc[:, "u-component_of_wind_isobaric"].mean()
myconfig['u-component_of_wind_isobaric_std'] = obs_train.loc[:, "u-component_of_wind_isobaric"].std()
myconfig['v-component_of_wind_isobaric_avg'] = obs_train.loc[:, "v-component_of_wind_isobaric"].mean()
myconfig['v-component_of_wind_isobaric_std'] = obs_train.loc[:, "v-component_of_wind_isobaric"].std()

obs_train = normalize_observations(obs_train)
actions_train = normalize_actions(actions_train)
obs_test = normalize_observations(obs_test)
actions_test = normalize_actions(actions_test)

x_train = obs_train.values
x_test = obs_test.values
y_train = actions_train.values
y_test = actions_test.values

#x_train = obs_train
#x_test = obs_test
#y_train = actions_train
#y_test = actions_test

# TensorFlow Dataset
datas = Dataset(x_train, x_test, y_train, y_test)

def train():
    # Setup Encoder

    temperature = tf.placeholder(tf.float64, [], name='temperature') #tf.float32 !!!...
    learning_rate = tf.placeholder(tf.float64, [], name='lr_value')
    inputs = tf.placeholder(tf.float64, shape=[None, 10], name='enc_inp')
    y_holder = tf.placeholder(tf.float64, shape=[None, 3], name='y_holder')
    #latent = tf.placeholder(tf.float64, shape=[None, 5], name='latent')
    latent = tf.placeholder(tf.float64, shape=[None, 3], name='latent')

    #initial_latent = [1., 0., 0.]
    initial_latent = [1., 0., 0.]
    initial_latent = np.asarray(initial_latent)
    initial_latent = np.expand_dims(initial_latent, axis=0)

    logits_y, q_y, log_q_y, encod_model, enc_inpp, latents, latents_shape, encc_shape = encoder(inputs, latent, temperature)

    # Setup Decoder
    decoder_last_layer, softmax, dec_model, new_inp = decoder(inputs, latents)

    # Operations
    train_op, loss = create_train_op(y_holder, learning_rate, decoder_last_layer) # NO_KL
    #train_op, loss, tmp, kl_tmp, KL, elbo = create_train_op_kl(y_holder, decoder_last_layer, learning_rate, q_y, log_q_y) # WITH KL

    # Initialize Session
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session(config=config)
    saver = tf.train.Saver(encod_model.weights) #TI KANOUME SAVE
    #saver.restore(sess, "./results_gumbel_softmax/checkpoint/run5/encoder/encoder2000/encoder_model_e1999-i49999.ckpt")
    dec_saver = tf.train.Saver(dec_model.weights)  # TI KANOUME SAVE
    #dec_saver.restore(sess, "./results_gumbel_softmax/checkpoint/run5/decoder/encoder2000/decoder_model_e1999-i49999.ckpt")
    #K.set_session(sess)
    sess.run(init_op)

    dat = []

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #ploti = []
    plote = []
    #plot_latent_var = []

    len1, len2 = datas.count_train_num_examples()
    x_test_len, y_test_len = datas.count_test_num_examples()

    global previous_latent
    global previous_latent_test
    global counter
    counter = 1
    latent_flag = True
    vae_obs = []
    vae_actions = []
    losses = []
    losses_test = []

    ploti = []
    for i in range(327072):
        ploti.append(i)

    try:
        for e in range(FLAGS.epochs):#for e in range(601, FLAGS.epochs):
            counter = 1
            losses_per_epoch = []
            losses_per_epoch_test = []
            #ploti = []
            plot_latent_var = []
            plote.append(e)
            print(e, "/1999")
            for i in range(0, 327072, 32): #1563x32, 782x64
                #print('i:', i)
                #Get Next Batch
                np_x, np_y = datas.train_next_batch(i)
                np_x_test, np_y_test = datas.test_next_batch(i)
                np_x = np.asarray(np_x)
                np_y = np.asarray(np_y)

                #Get Test Next Batch
                np_x_test = np.asarray(np_x_test)
                np_y_test = np.asarray(np_y_test)

                #if (counter % 1000 == 0 or counter == 1):
                if (counter == 1):
                    latent_flag = True
                    #print('counter:', counter, '\ni:', i, 'flag:', latent_flag)
                else:
                    latent_flag = False

                #WITH_KL
                #_, np_loss = sess.run([train_op, loss], {enc_inputs: input_enc_batch2, y_holder: y_batch, dec_inputs: input_dec_batch, learning_rate: FLAGS.learning_rate, temperature: FLAGS.init_temp})
                #NO_KL
                if (latent_flag == True):
                    #print('hey')
                    #tmp_sess, kl_tmp_sess, KL_sess, elbo_sess, _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run([tmp, kl_tmp, KL, elbo, train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                    _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run([train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp,latents, encc_shape],
                                                                      {inputs: np_x, latent: initial_latent,
                                                                       y_holder: np_y,
                                                                       learning_rate: FLAGS.learning_rate,
                                                                       temperature: FLAGS.init_temp})
                    encoder_sess = np.asarray(latent_sess)

                else:
                    #print('hey2')
                    #tmp_sess, kl_tmp_sess, KL_sess, elbo_sess, _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run([tmp, kl_tmp, KL, elbo, train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                    _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run([train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp,latents, encc_shape],
                                                                      {inputs: np_x, latent: encoder_sess,
                                                                       y_holder: np_y,
                                                                       learning_rate: FLAGS.learning_rate,
                                                                       temperature: FLAGS.init_temp})
                    encoder_sess = np.asarray(latent_sess)
                #print('train_loss:', np_loss)
                losses_per_epoch.append(np_loss)
                #losses_per_epoch_test.append(np_loss_test)

                list=[]
                for j in range(32):
                    #print('latent:', encoder_sess[j])
                    encoder_sess_plot = np.argmax(encoder_sess[j])
                    #list.append(encoder_sess_plot)
                    #print('list:', list)
                    #print('plot:', encoder_sess_plot)
                    plot_latent_var.append(encoder_sess_plot)

                encoder_sess = encoder_sess[31]
                encoder_sess = np.expand_dims(encoder_sess, 0)
                #print('EEEE:', encoder_sess)

                counter += 1

                if(i>=327040):
                    losses_per_epoch_np = np.asarray(losses_per_epoch)
                    mean_losses_per_epoch = np.mean(losses_per_epoch_np)
                    print('train_loss:', mean_losses_per_epoch)
                    losses.append(mean_losses_per_epoch.tolist())

                    #losses_per_epoch_test_np = np.asarray(losses_per_epoch_test)
                    #mean_losses_per_epoch_test = np.mean(losses_per_epoch_test_np)
                    #losses_test.append(mean_losses_per_epoch_test.tolist())

                    ###########################################################################

                ## if i % 10000 == 1:
                if e % 100 == 0 and (i >=327040):
                    encoder_path = saver.save(sess, FLAGS.checkpoint_dir + '/encoder/encoder_test/encoder_model_e{}-i{}.ckpt'.format(e, i))
                    decoder_path = dec_saver.save(sess, FLAGS.checkpoint_dir + '/decoder/decoder_test/decoder_model_e{}-i{}.ckpt'.format(e, i))
                if e == 999 and (i >=327040):#(i % 1000 == 0 or i == 49999):
                    encoder_path = saver.save(sess, FLAGS.checkpoint_dir + '/encoder/encoder1000/encoder_model_e{}-i{}.ckpt'.format(e, i))
                    decoder_path = dec_saver.save(sess, FLAGS.checkpoint_dir + '/decoder/decoder1000/decoder_model_e{}-i{}.ckpt'.format(e, i))
                elif e == 1999 and (i >=327040):
                    encoder_path = saver.save(sess, FLAGS.checkpoint_dir + '/encoder/encoder2000/encoder_model_e{}-i{}.ckpt'.format(e, i))
                    decoder_path = dec_saver.save(sess, FLAGS.checkpoint_dir + '/decoder/decoder2000/decoder_model_e{}-i{}.ckpt'.format(e, i))

                #ploti.append(i)


                if (e % 100 == 0 and i >= 327040) or (e == 999 and i >= 327040) or (e == 1999 and i >= 327040):
                    plot_vae(ploti, plot_latent_var, e)
                    #plot_both_loss(plote, losses,losses_test, e)
                    plot_loss(plote, losses, e)

            FLAGS.init_temp = np.maximum(FLAGS.init_temp * np.exp(-FLAGS.anneal_rate * e), FLAGS.min_temp)
            #FLAGS.learning_rate *= 0.9
            print('Temperature updated to {}\n'.format(FLAGS.init_temp))

    except KeyboardInterrupt:
        print()

    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

        print('Mean of Losses:', np.mean(losses))
        print('x_train_num::', len1, 'y_train:', len2, 'x_test_num:', x_test_len, 'y_test_num', y_test_len)
        #plot_vae(ploti, plot_latent_var)


def plot_vae(ploti, latent_var, e):
    ploti = np.asarray(ploti)
    latent_var = np.asarray(latent_var)
    e = np.asarray(e)
    for i in range(0, 327072, 1000):
        # print(i)
        if (i == 0):
            ploti2 = ploti[:i + 1000].copy()
            save_path = os.getcwd() + '/results_gumbel_softmax/plot/aviation/epoch{}-i{}_aviation_latents.png'.format(e, i)
            pred_context = latent_var[:i + 1000].copy()
        else:
            ploti2 = ploti[j:i + 1000].copy()
            save_path = os.getcwd() + '/results_gumbel_softmax/plot/aviation/epoch{}-i{}_aviation_latents.png'.format(e, i)
            pred_context = latent_var[j:i + 1000].copy()

        j = i + 1000
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        #ax.scatter(ploti2, pred_context, c='r', marker='X')
        ax.plot(ploti2, pred_context)
        plt.xlabel("TimeSteps")
        # plt.xticks([0, 1000 ,3000, 5000, 7000, 9000, 9500, 10000])
        # naming the y axis
        plt.ylabel("Latent Vars")
        # giving a title to my graph
        plt.title("Aviation - Barcelona to Madrid")
        plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
        plt.clf()
        plt.close(fig)

def plot_vae_test(ploti, latent_var, e):
    ploti = np.asarray(ploti)
    latent_var = np.asarray(latent_var)
    ploti_size = ploti.size
    latent_var_size = latent_var.size
    e = np.asarray(e)

    ploti = ploti[:ploti_size].copy()
    save_path = os.getcwd() + '/results_gumbel_softmax/plot/aviation/epoch{}_aviation_latents.png'.format(e)
    pred_context = latent_var[:ploti_size].copy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(ploti, pred_context, c='r', marker='X')
    ax.plot(ploti, pred_context)
    plt.xlabel("TimeSteps")
    #plt.xticks([0, 1000 ,3000, 5000, 7000, 9000, 9500, 10000])
    # naming the y axis
    plt.ylabel("Latent Vars")
    # giving a title to my graph
    plt.title("Aviation - Barcelona to Madrid")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, e):
    epochs = np.asarray(epochs)
    loss = np.asarray(loss)
    e = np.asarray(e)

    epochs = epochs.copy()
    save_path = os.getcwd() + '/results_gumbel_softmax/plot/aviation/epoch{}_aviation_loss.png'.format(e)
    pred_context = loss.copy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(epochs, pred_context, c='r', marker='X')
    ax.plot(epochs, pred_context)
    plt.xlabel("Epochs")
    #plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # naming the y axis
    plt.ylabel("Loss")
    # giving a title to my graph
    plt.title("Aviation - Barcelona to Madrid")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_both_loss(epochs, loss, loss_test, e):
    epochs = np.asarray(epochs)
    loss = np.asarray(loss)
    loss_test = np.asarray(loss_test)
    e = np.asarray(e)

    epochs = epochs.copy()
    save_path = os.getcwd() + '/results_gumbel_softmax/plot/aviation/epoch{}_aviation_loss.png'.format(e)
    pred_context = loss.copy()
    pred_context2 = loss_test.copy()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(epochs, pred_context, c='r', marker='X')
    ax.plot(epochs, pred_context, label="line 1", color="cornflowerblue")
    ax2.plot(epochs, pred_context2, label="line 2", color="Orange")
    ax.set_xlabel('Epochs')
    #plt.xlabel("Epochs")
    #plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # naming the y axis
    #plt.ylabel("Loss")
    ax.set_ylabel('Loss', color="cornflowerblue")
    ax2.set_ylabel('Test Loss', color="Orange")
    # giving a title to my graph
    plt.title("Aviation - Barcelona to Madrid")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def renameVAE():
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    # RENAME MAP LAYERS-BIASES-GRAPHS
    OLD_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/encoder/encoder2000/encoder_model_e1999-i327040.ckpt'  # 327040 #'/encoder/encoder_model49599.ckpt'
    NEW_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/test/encoder/encoder_model_e2000.ckpt'  # '/encoder_new(2)/encoder_model_new49600.ckpt'
    # OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model9999.ckpt'
    # NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model_new/model_new9999.ckpt'

    reader = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE)
    shape_from_key = reader.get_variable_to_shape_map()
    dtype_from_key = reader.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')
    # print(dtype_from_key)

    vars_to_rename = {
        "encoder_h1/kernel": "Encoder/dense/kernel",
        "encoder_h1/bias": "Encoder/dense/bias",
        "encoder_h2/kernel": "Encoder/dense_1/kernel",
        "encoder_h2/bias": "Encoder/dense_1/bias",
        "encoder_out/bias": "Encoder/dense_2/bias",
        "encoder_out/kernel": "Encoder/dense_2/kernel",

        # "_CHECKPOINTABLE_OBJECT_GRAPH": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader1 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader1.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader1.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)

    # Read New_Checkpoint
    reader2 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE)
    shape_from_key2 = reader2.get_variable_to_shape_map()
    dtype_from_key2 = reader2.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

    OLD_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/decoder/decoder2000/decoder_model_e1999-i327040.ckpt'  # 1999-49999.ckpt'
    NEW_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/test/decoder/decoder_model_e2000.ckpt'  # 2000_new50000.ckpt'

    reader2 = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE2)
    shape_from_key = reader2.get_variable_to_shape_map()
    dtype_from_key = reader2.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')

    vars_to_rename = {
        "decoder_h1/kernel": "Policy/dense_3/kernel",
        "decoder_h1/bias": "Policy/dense_3/bias",
        "decoder_h2/kernel": "Policy/dense_4/kernel",
        "decoder_h2/bias": "Policy/dense_4/bias",
        "decoder_out/bias": "Policy/dense_5/bias",
        "decoder_out/kernel": "Policy/dense_5/kernel",

    }
    new_checkpoint_vars = {}
    reader3 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE2)
    for old_name in reader3.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader3.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver2 = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver2.save(sess, NEW_CHECKPOINT_FILE2)

    # Read New_Checkpoint
    reader4 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE2)
    shape_from_key2 = reader4.get_variable_to_shape_map()
    dtype_from_key2 = reader4.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

def rename():
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    # RENAME MAP LAYERS-BIASES-GRAPHS
    OLD_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/encoder/encoder2000/encoder_model_e1999-i327040.ckpt'  # '/encoder/encoder_model49599.ckpt'
    NEW_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/trpo_plugins/encoder/encoder_model_e2000.ckpt'  # '/encoder_new(2)/encoder_model_new49600.ckpt'
    # OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model9999.ckpt'
    # NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model_new/model_new9999.ckpt'

    reader = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE)
    shape_from_key = reader.get_variable_to_shape_map()
    dtype_from_key = reader.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')
    # print(dtype_from_key)

    vars_to_rename = {
        "encoder_h1/kernel": "Encoder/dense_6/kernel",
        "encoder_h1/bias": "Encoder/dense_6/bias",
        "encoder_h2/kernel": "Encoder/dense_7/kernel",
        "encoder_h2/bias": "Encoder/dense_7/bias",
        "encoder_out/bias": "Encoder/dense_8/bias",
        "encoder_out/kernel": "Encoder/dense_8/kernel",

        # "_CHECKPOINTABLE_OBJECT_GRAPH": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader1 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader1.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader1.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)

    # Read New_Checkpoint
    reader2 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE)
    shape_from_key2 = reader2.get_variable_to_shape_map()
    dtype_from_key2 = reader2.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

    OLD_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/decoder/decoder2000/decoder_model_e1999-i327040.ckpt'  # 1999-49999.ckpt'
    NEW_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run2(batch-32-modes3)' + '/trpo_plugins/decoder/decoder_model_e2000.ckpt'  # 2000_new50000.ckpt'

    reader2 = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE2)
    shape_from_key = reader2.get_variable_to_shape_map()
    dtype_from_key = reader2.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')

    vars_to_rename = {
        "decoder_h1/kernel": "Policy/dense_9/kernel",
        "decoder_h1/bias": "Policy/dense_9/bias",
        "decoder_h2/kernel": "Policy/dense_10/kernel",
        "decoder_h2/bias": "Policy/dense_10/bias",
        "decoder_out/bias": "Policy/dense_11/bias",
        "decoder_out/kernel": "Policy/dense_11/kernel",

    }
    new_checkpoint_vars = {}
    reader3 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE2)
    for old_name in reader3.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader3.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver2 = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver2.save(sess, NEW_CHECKPOINT_FILE2)

    # Read New_Checkpoint
    reader4 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE2)
    shape_from_key2 = reader4.get_variable_to_shape_map()
    dtype_from_key2 = reader4.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

#TESTING

class ClonosEncoder(object):
    @staticmethod
    def create_encoder(observation_dimensions, latent_dimensions):
        with tf.name_scope('Encoder'):
            encoder_x = Input(shape=(observation_dimensions+latent_dimensions,), dtype=tf.float64)

            l1 = Dense(100, activation='tanh')(encoder_x) #name='dense_6'), activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            l2 = Dense(100, activation='tanh')(l1) #name='dense_7')

            out = Dense(latent_dimensions, activation='softmax')(l2) #name='dense_8')

        model = Model(inputs=encoder_x, outputs=out)

        return model, encoder_x, l1, l2


class Policy(object):
    @staticmethod
    def create_policy(observation_dimensions, latent_dimensions, action_dimensions):
        with tf.name_scope('Policy'):
            x = Input(shape=(observation_dimensions+latent_dimensions,), dtype=tf.float64)

            h = Dense(100, activation='tanh')(x)#, name='Policy/dense_9')(x)
            h1 = Dense(100, activation='tanh')(h)#, name='Policy/dense_10')(h)

            out = Dense(action_dimensions)(h1)#, name='Policy/dense_11')(h1)

        model = Model(inputs=x, outputs=out)

        return model, x, h, h1

class Agent(object):
    def __init__(self, observation_dimensions=10, latent_dimensions=3, action_dimensions=3):#latent_dimensions=5
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.latent_dimensions = latent_dimensions
        self.sess = tf.Session(config=config)

        self.model2, self.encoder_x, self.l1, self.l2 = ClonosEncoder.create_encoder(self.observation_dimensions,self.latent_dimensions)
        self.model, self.x, self.h, self.h1 = Policy.create_policy(self.observation_dimensions, self.latent_dimensions,self.action_dimensions)

        self.encoder_logits = self.model2.outputs[0]
        self.logits = self.model.outputs[0]
        self.env = environment.Environment()
        self.sess.run(tf.global_variables_initializer())

    def one_hot_encoding(self, x):
        # print('OneHotEncoding...')
        argmax = np.argmax(x)
        self.plot_test_latent_var.append(argmax)
        encoded = to_categorical(argmax, num_classes=3)#5)
        # print('enc:', encoded)
        return encoded.tolist(), argmax

    def init_encoder(self, observation):
        init_latent = [1., 0., 0.]
        #init_latent = [1., 0., 0., 0., 0.]
        init_latent = np.asarray(init_latent)
        enc_input = np.concatenate((np.asarray(observation), init_latent))
        #print('obs_l:', enc_input)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
        latent_prob = np.asarray(mu2)
        latent, arg = self.one_hot_encoding(latent_prob)

        decoder_input = np.concatenate((np.asarray(observation), latent))
        return decoder_input, latent, arg

    def new_encoder(self, observation, latent_new):
        global global_concat_test

        if self.init == True:
            enc_input_n = np.concatenate((np.asarray(observation), latent_new))
            #print('obs_l:',enc_input_n)
        else:
            enc_input_n = np.concatenate((np.asarray(observation), global_concat_test))
            #print('obs_l:', enc_input_n)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input_n]})[0]

        latent_prob = np.asarray(mu2)
        latent_new, arg = self.one_hot_encoding(latent_prob)

        # array, shape = self.latent_sequence(mu2)

        # one_hot_enc = self.keras_oneHotEncoder(array, shape)

        # exit(0)

        global_concat_test = latent_new
        decoder_input = np.concatenate((np.asarray(observation), latent_new))
        return decoder_input, latent_new, arg

    def act_test(self, decoder_input):  # , latent):
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]
        act = mu
        return act

    def test(self):
        out_file = 'results_gumbel_softmax/checkpoint/run3(batch-32-modes5-nokl)/test/0_VAE_results.csv'
        action_out_file = 'results_gumbel_softmax/checkpoint/run3(batch-32-modes5-nokl)/test/0_actions_VAE_results.csv'

        enc_saver = tf.train.Saver(self.model2.weights)
        enc_saver.restore(self.sess, "results_gumbel_softmax/checkpoint/run3(batch-32-modes5-nokl)/test/encoder/encoder_model_e2000.ckpt")#run2(batch-32-modes3)
        saver = tf.train.Saver(self.model.weights)
        saver.restore(self.sess, "results_gumbel_softmax/checkpoint/run3(batch-32-modes5-nokl)/test/decoder/decoder_model_e2000.ckpt")#run2(batch-32-modes3)

        with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
            results.write("episode,longitude,latitude,altitude,timestamp,Pressure_surface,"
                          "Relative_humidity_isobaric,Temperature_isobaric,Wind_speed_gust_surface,"
                          "u-component_of_wind_isobaric,v-component_of_wind_isobaric,mode" +
                          "\n")
            action_results.write("episode,dlon,dlat,dalt\n")

            actions = []
            obs = []
            discounted_rewards = []
            total_rewards = []
            print('Episodes,Reward')
            self.init = True

            for i_episode in range(50):
                self.counter2 = 0
                self.plot_test_latent_var = []
                self.plot_i_test = []
                steps = 0
                r_i = []
                norm_observation, observation  = self.env.reset()
                obs_unormalized = unnormalize_observation(observation)
                print('obs:',observation)
                dec_input1, latent_init, arg = self.init_encoder(observation)

                line = ""
                for ob in obs_unormalized:
                    line += "," + str(ob)
                line += "," + str(arg)
                results.write(str(i_episode) + line + "\n")

                total_reward = 0

                for t in range(1000):
                    if self.init:
                        print('latent:', latent_init)
                        print('dec_input:', dec_input1)
                        obs.append(dec_input1)
                        action = self.act_test(dec_input1)
                        action = action.tolist()
                    else:
                        obs.append(dec_input2)
                        action = self.act_test(dec_input2)
                        action = action.tolist()

                    self.counter2 += 1

                    norm_observation2, observation2, reward, done = self.env.step(action, t)
                    #print('obs2:',observation2)

                    if self.init:
                        dec_input2, latent, arg = self.new_encoder(observation2, latent_init)
                        #print('latent:',latent)
                        #print('dec_in2:',dec_input2)
                    else:
                        dec_input2, latent, arg = self.new_encoder(observation2, latent)
                        #print('latent_2:',latent)
                        #print('dec_in2_2:',dec_input2)

                    steps += 1
                    r_i.append(reward)
                    actions.append(action)
                    total_reward += reward
                    action = unnormalize_action(action)
                    obs_unormalized2 = unnormalize_observation(observation2)

                    self.plot_i_test.append(t)
                    self.init = False
                    if t % 100 == 0:
                        print("%i/%i" % (t+100, 1000))
                    if t >= 1000 or done:
                        #if done:
                        #    print('')
                        #exit(0)
                        #continue

                        np1 = np.asarray(self.plot_i_test)
                        np2 = np.asarray(self.plot_test_latent_var)
                        print('np1_s:', np1.size)
                        print('np2_s:', np2.size)
                        plot_vae_test(self.plot_i_test, self.plot_test_latent_var, i_episode)
                        break
                    else:
                        line = ""
                        for ob in obs_unormalized2:
                            line += "," + str(ob)
                        line += "," + str(arg)
                        results.write(str(i_episode) + line +"\n")
                    action_results.write(str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," +str(action[2]) + "\n")

                print('{0},{1}'.format(i_episode, total_reward))
                #exit(0)
                discounted_rewards.extend(discount(r_i, 0.995))
                total_rewards.append(total_reward)


                #self.env.close()
                #self.sess.close()
            return actions, obs, discounted_rewards, total_rewards


def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.data_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    tf.gfile.MakeDirs(FLAGS.results_dir)

    train()
    print("Vae model trained")

    #RENAME LAYERS
    renameVAE()
    rename()

    # Testing
    agent = Agent()
    agent.env.read_starting_points(random_choice=False, fname='0%_test')
    actions_vae, obs_vae, discounted_rewards_vae, total_rewards_vae  = agent.test()
    print('Sum of Rewards VAE:', sum(total_rewards_vae))  # np.mean(total_rewards_vae)
    print('Mean Reward VAE:', np.mean(total_rewards_vae))



if __name__=="__main__":
    main()
