# from myconfig import myconfig
from myconfig import myconfig
import tensorflow as tf
import copy
from tensorflow.python.keras.layers import Input, Dense, concatenate, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mean_squared_error
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from tensorflow.keras.utils import to_categorical
from collections import deque
# import warnings
from critic import Critic
from utils import conjugate_gradient, set_from_flat, kl, self_kl, \
    flat_gradient, get_flat, discount, line_search, gauss_log_prob, visualize, gradient_summary, \
    unnormalize_action, unnormalize_observation, unnormalize_observation_metar, unnormalize_observation2
import random


# from cartpole.critic.critic import Critic
# from cartpole.trpo.utils import conjugate_gradient, set_from_flat, kl, self_kl,\
#     flat_gradient, get_flat, discount, line_search
# np.seterr(all='warn')
# warnings.filterwarnings('error')

# http://rail.eecs.berkeley.edu/deeprlcoursesp17/docs/lec5.pdf

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.6


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
        """
        Creates the model of the policy.
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        :return: Model and the Input layer.
        """
        with tf.name_scope('Policy'):
            x = Input(shape=(observation_dimensions+latent_dimensions,), dtype=tf.float64)

            h = Dense(100, activation='tanh')(x)#, name='Policy/dense_9')(x)
            h1 = Dense(100, activation='tanh')(h)#, name='Policy/dense_10')(h)

            out = Dense(action_dimensions)(h1)#, name='Policy/dense_11')(h1)

        model = Model(inputs=x, outputs=out)

        return model, x, h, h1


class Discriminator(object):

    def __init__(self, observation_dimensions=4, action_dimensions=2):
        self.alpha = myconfig['discriminator_alpha']
        self.epochs = myconfig['discriminator_epochs']
        self.sess = tf.Session(config=config)

        self.observations_input = Input(shape=(observation_dimensions,), dtype=tf.float64)
        self.actions_input = Input(shape=(action_dimensions,), dtype=tf.float64)
        self.input = concatenate([self.observations_input, self.actions_input])

        h1 = Dense(100, activation='tanh')(self.input)
        h2 = Dense(100, activation='tanh')(h1)

        self.out = Dense(1)(h2)

        self.discriminate = Model(inputs=[self.observations_input, self.actions_input],
                                  outputs=self.out)

        self.log_D = tf.log(tf.nn.sigmoid(self.out))

        self.expert_samples_observations = Input(shape=(observation_dimensions,),
                                                 dtype=tf.float64)
        self.expert_samples_actions = Input(shape=(action_dimensions,), dtype=tf.float64)
        self.policy_samples_observations = Input(shape=(observation_dimensions,), dtype=tf.float64)
        self.policy_samples_actions = Input(shape=(action_dimensions,), dtype=tf.float64)
        self.expert_samples_out = self.discriminate([self.expert_samples_observations,
                                                     self.expert_samples_actions])
        self.policy_samples_out = self.discriminate([self.policy_samples_observations,
                                                     self.policy_samples_actions])

        self.discriminator = Model(inputs=[self.expert_samples_observations,
                                           self.expert_samples_actions,
                                           self.policy_samples_observations,
                                           self.policy_samples_actions
                                           ],
                                   outputs=[self.expert_samples_out,
                                            self.policy_samples_out])

        # self.expert_loss = tf.reduce_mean(tf.logs(tf.ones_like(self.expert_samples_out)-tf.nn.sigmoid(self.expert_samples_out)))
        # self.policy_loss = tf.reduce_mean(tf.logs(tf.nn.sigmoid(self.expert_samples_out)))
        # self.loss = -(self.expert_loss + self.policy_loss)
        self.expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.expert_samples_out,
                                                                   labels=tf.zeros_like(
                                                                       self.expert_samples_out))
        self.policy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.policy_samples_out,
                                                                   labels=tf.ones_like(
                                                                       self.policy_samples_out))
        self.expert_loss_avg = tf.reduce_mean(self.expert_loss)
        self.policy_loss_avg = tf.reduce_mean(self.policy_loss)
        self.loss = tf.reduce_mean(self.expert_loss) + tf.reduce_mean(self.policy_loss)

        # self.predictions = self.discriminate.outputs[0]
        # self.labels = tf.placeholder(tf.float32, shape=(None), name='y')
        # self.loss = tf.nn.sigmoid.cross_entropy_with_logits(self.out,self.labels)
        self.opt = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        # self.opt = tf.train.RMSPropOptimizer(self.alpha, decay=0.9).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        with open(myconfig['output_dir']+'/disc_train_loss_log.csv', 'w') as disc_train_log:
            disc_train_log.write("epoch,total_loss,expert_loss,policy_loss"+"\n")


    def get_trainable_weights(self):
        return self.sess.run(
                [self.discriminate.trainable_weights], feed_dict={})[0]

    def train(self, expert_samples_observations, expert_samples_actions,
              policy_samples_observations, policy_samples_actions):
        with open(myconfig['output_dir']+'/disc_train_loss_log.csv', 'a') as disc_train_log:
            loss_run = 0
            loss_before_train = 0
            for i in range(self.epochs):
                _, loss_run, expert_loss_run, policy_loss_run = self.sess.run([self.opt, self.loss, self.expert_loss_avg,self.policy_loss_avg],
                                            feed_dict={
                                                self.expert_samples_observations:
                                                    expert_samples_observations,
                                                self.expert_samples_actions:
                                                    expert_samples_actions,
                                                self.policy_samples_observations:
                                                    policy_samples_observations,
                                                self.policy_samples_actions:
                                                    policy_samples_actions
                                            })
                if i == 0:
                    loss_before_train = loss_run

                disc_train_log.write(str(i) + "," + str(loss_run)+"," + str(expert_loss_run) + "," + str(policy_loss_run)+"\n")
                # print('Discriminator loss:', loss_run)
                # if i % 100 == 0: print(i, "loss:", loss_run)
        return loss_before_train, loss_run

    def predict(self, samples_observations, samples_actions):
        return self.sess.run(self.log_D,
                             feed_dict={self.observations_input: samples_observations,
                                        self.actions_input: samples_actions})




class TRPOAgent(object):

    def __init__(self, env, observation_dimensions=10, latent_dimensions=3, action_dimensions=3):
        """
        Initializes the agent's parameters and constructs the flowgraph.
        :param env: Environment
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        """
        self.latent_list = []
        self.latent_sequence1 = []
        self.latent_sequence_prob = []
        self.encoder_rew = []
        self.encoder_rew_reset = []
        self.counter = 0
        self.counter2 = 0
        self.init = True

        self.env = env
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.latent_dimensions = latent_dimensions
        self.path_size = myconfig['path_size']
        self.mini_batch_size = myconfig['mini_batch_size']
        self.mini_batches = myconfig['mini_batches']
        self.gamma = myconfig['gamma']
        self.lamda = myconfig['lamda']
        self.max_kl = myconfig['max_kl']
        self.total_episodes = 0
        self.logstd = np.float64(myconfig['logstd'])
        self.critic = Critic(observation_dimensions=self.observation_dimensions)
        self.discriminator = Discriminator(observation_dimensions=self.observation_dimensions, action_dimensions=self.action_dimensions)
        # self.replay_buffer = ReplayBuffer()
        self.sess = tf.Session(config=config)
        self.model2, self.encoder_x, self.l1, self.l2 = ClonosEncoder.create_encoder(self.observation_dimensions, self.latent_dimensions)
        self.model, self.x, self.h, self.h1 = Policy.create_policy(self.observation_dimensions, self.latent_dimensions, self.action_dimensions)

        visualize(self.model.trainable_weights)

        self.episode_history = deque(maxlen=100)

        self.advantages_ph = tf.placeholder(tf.float64, shape=None)
        self.actions_ph = tf.placeholder(tf.float64, shape=(None, action_dimensions),)
        self.old_log_prob_ph = tf.placeholder(tf.float64, shape=None)
        self.theta_ph = tf.placeholder(tf.float64, shape=None)
        self.tangent_ph = tf.placeholder(tf.float64, shape=None)
        self.mu_old_ph = tf.placeholder(tf.float64, shape=(None, action_dimensions))

        self.encoder_logits = self.model2.outputs[0]
        self.logits = self.model.outputs[0]

        #EDW
        self.q = self.encoder_logits
        self.argmax_q = tf.argmax(self.q, axis=1)
        # self.log_q = tf.log(self.argmax_q)

        var_list = self.model.trainable_weights
        self.flat_vars = get_flat(var_list)
        self.sff = set_from_flat(self.theta_ph, var_list)

        self.step_direction = tf.placeholder(tf.float64, shape=None)
        self.g_sum = gradient_summary(self.step_direction, var_list)

        # Compute surrogate.
        self.log_prob = gauss_log_prob(self.logits, self.logstd, self.actions_ph)
        neg_lh_divided = tf.exp(self.log_prob - self.old_log_prob_ph)
        w_neg_lh = neg_lh_divided * self.advantages_ph
        self.surrogate = tf.reduce_mean(w_neg_lh)

        kl_op = kl(self.logits, self.logstd, self.mu_old_ph, self.logstd)
        self.losses = [self.surrogate, kl_op]

        self.flat_grad = flat_gradient(self.surrogate, var_list)
        # Compute fisher vector product
        self_kl_op = self_kl(self.logits, self.logstd)
        self_kl_flat_grad = flat_gradient(self_kl_op, var_list)
        g_vector_dotproduct = tf.reduce_sum(self_kl_flat_grad * self.tangent_ph)
        # self.self_kl_grad = tf.gradients(self_kl_op, var_list)
        # start = 0
        # tangents = []
        # for var in var_list:
        #     end = start+np.prod(var.shape)
        #     tangents.append(tf.reshape(tangent_ph[start:end],var.shape))
        #     start = end
        # g_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(
        # self_kl_grad, tangents)]
        self.fvp = flat_gradient(g_vector_dotproduct, var_list)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(myconfig['log_dir'], self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def predict(self, samples_observations):
        q_sess, q_arg = self.sess.run([self.q, self.argmax_q], feed_dict={self.encoder_x: samples_observations})
        #print('q: \n', q_sess)
        #print('q_argmax:\n', q_arg)
        #print('len_q:', len(q_sess), 'len_q_arg2:', len(q_arg))

        return q_sess, q_arg

    def __fisher_vector_product(self, g, feed):
        """
        Computes fisher vector product H*g using the direct method.
        :param p: Gradient of surrogate g.
        :param feed: Dictionary, feed_dict for tf.placeholders.
        :return: Fisher vector product H*g.
        """
        damping = myconfig['fvp_damping']
        feed[self.tangent_ph] = g
        fvp_run = self.sess.run(self.fvp, feed)
        assert fvp_run.shape == g.shape, "Different shapes. fvp vs g"
        return fvp_run + g * damping

    def get_vars(self):
        model2_weights = self.model2.weights
        model_weights = self.model.weights
        return model2_weights, model_weights

    def encoder_rewards(self, prob_latents):
        winner = np.argmax(prob_latents)
        encoder_reward = np.log(prob_latents[winner])
        #print('r:',encoder_reward)
        self.encoder_rew.append([encoder_reward])
        #array = np.asarray(self.encoder_rew[:])

    def one_hot_encoding(self, x):
        # print('OneHotEncoding...')
        argmax = np.argmax(x)
        encoded = to_categorical(argmax, num_classes=3)
        # print('enc:', encoded)
        return encoded.tolist(), argmax

    def test_one_hot_encoding(self, x):
        # print('OneHotEncoding...')
        argmax = np.argmax(x)
        self.plot_test_latent_var.append(argmax)
        encoded = to_categorical(argmax, num_classes=3)
        # print('enc:', encoded)
        return encoded.tolist(), argmax

    def act(self, observation, latent_seq):  # , latent):
        global global_concat
        global obs_matrix

        #print('self.counter = ', self.counter)
        decoder_input = np.concatenate((np.asarray(observation), latent_seq[self.counter]))
        obs_matrix = decoder_input
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]

        act = mu + self.logstd * np.random.randn(self.action_dimensions)
        self.counter += 1

        return act, mu  # , log_q #m2

    def init_encoder(self, observation):
        init_latent = [1., 0., 0.]
        # init_latent = [1., 0., 0., 0., 0.]
        init_latent = np.asarray(init_latent)
        enc_input = np.concatenate((np.asarray(observation), init_latent))
        # print('obs_l:', enc_input)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
        latent_prob = np.asarray(mu2)
        latent, arg = self.test_one_hot_encoding(latent_prob)

        decoder_input = np.concatenate((np.asarray(observation), latent))
        return decoder_input, latent, arg

    def new_encoder(self, observation, latent_new):
        global global_concat_test

        if self.init == True:
            enc_input_n = np.concatenate((np.asarray(observation), latent_new))
            # print('obs_l:',enc_input_n)
        else:
            enc_input_n = np.concatenate((np.asarray(observation), global_concat_test))
            # print('obs_l:', enc_input_n)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input_n]})[0]

        latent_prob = np.asarray(mu2)
        latent_new, arg = self.test_one_hot_encoding(latent_prob)

        # array, shape = self.latent_sequence(mu2)

        # one_hot_enc = self.keras_oneHotEncoder(array, shape)

        global_concat_test = latent_new
        decoder_input = np.concatenate((np.asarray(observation), latent_new))
        return decoder_input, latent_new, arg

    def act_test(self, decoder_input):#, latent):
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]
        act = mu
        return act

    def plot_vae_test(self, ploti, latent_var, e, savepath):
        ploti = np.asarray(ploti)
        latent_var = np.asarray(latent_var)
        ploti_size = ploti.size
        latent_var_size = latent_var.size
        e = np.asarray(e)

        ploti = ploti[:ploti_size].copy()
        save_path = os.getcwd() + savepath.format(e)#'/VAE/results_gumbel_softmax/plot/aviation/epoch{}_aviation_latents.png'
        pred_context = latent_var[:ploti_size].copy()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(ploti, pred_context, c='r', marker='X')
        ax.plot(ploti, pred_context)
        plt.xlabel("TimeSteps")
        # plt.xticks([0, 1000 ,3000, 5000, 7000, 9000, 9500, 10000])
        # naming the y axis
        plt.ylabel("Latent Vars")
        # giving a title to my graph
        plt.title("Aviation Enriched - Barcelona to Madrid")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def run(self, episode_num, vae=False, bcloning=False, fname='0%_validate'):
        if bcloning:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_bcloning_results.csv'
            action_out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_actions_bcloning_results.csv'
        elif vae:
            out_file = '../../../VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/0_VAE_results-Metar(3modes).csv'
            action_out_file = '../../../VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/0_actions_VAE_results-Metar(3modes).csv'

            enc_saver = tf.train.Saver(self.model2.weights)
            enc_saver.restore(self.sess,"VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")  # run2(batch-32-modes3)
            saver = tf.train.Saver(self.model.weights)
            saver.restore(self.sess,"VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/decoder/decoder_model_e2000.ckpt")  # run2(batch-32-modes3)

            with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
                results.write("episode,longitude,latitude,altitude,timestamp,Pressure_surface,"
                              "Relative_humidity_isobaric,Temperature_isobaric,Wind_speed_gust_surface,"
                              "u-component_of_wind_isobaric,v-component_of_wind_isobaric,drct,sknt,alti,vsby,gust,mode" +
                              "\n")
                action_results.write("episode,dlon,dlat,dalt\n")

                actions = []
                obs = []
                discounted_rewards = []
                total_rewards = []
                print('Episodes,Reward')
                self.init = True

                for i_episode in range(episode_num):
                    self.counter2 = 0
                    self.plot_test_latent_var = []
                    self.plot_i_test = []
                    steps = 0
                    r_i = []
                    norm_observation, observation = self.env.reset()
                    obs_unormalized = unnormalize_observation_metar(observation)
                    #print('obs:', observation)
                    dec_input1, latent_init, arg = self.init_encoder(observation)

                    line = ""
                    for ob in obs_unormalized:
                        line += "," + str(ob)
                    line += "," + str(arg)
                    results.write(str(i_episode) + line + "\n")

                    total_reward = 0

                    for t in range(1000):
                        if self.init:
                            #print('latent:', latent_init)
                            #print('dec_input:', dec_input1)
                            obs.append(dec_input1)
                            action = self.act_test(dec_input1)
                            action = action.tolist()
                        else:
                            obs.append(dec_input2)
                            action = self.act_test(dec_input2)
                            action = action.tolist()

                        self.counter2 += 1

                        norm_observation2, observation2, reward, done = self.env.step(action, t)
                        # print('obs2:',observation2)

                        if self.init:
                            dec_input2, latent, arg = self.new_encoder(observation2, latent_init)
                            # print('latent:',latent)
                            # print('dec_in2:',dec_input2)
                        else:
                            dec_input2, latent, arg = self.new_encoder(observation2, latent)
                            # print('latent_2:',latent)
                            # print('dec_in2_2:',dec_input2)

                        steps += 1
                        r_i.append(reward)
                        actions.append(action)
                        total_reward += reward
                        action = unnormalize_action(action)
                        obs_unormalized2 = unnormalize_observation_metar(observation2)

                        self.plot_i_test.append(t)
                        self.init = False
                        if t % 100 == 0:
                            print("%i/%i" % (t + 100, 1000))
                        if t >= 1000 or done:
                            # if done:
                            #    print('')
                            # exit(0)
                            # continue

                            np1 = np.asarray(self.plot_i_test)
                            np2 = np.asarray(self.plot_test_latent_var)
                            print('np1_s:', np1.size)
                            print('np2_s:', np2.size)
                            #self.plot_vae_test(self.plot_i_test, self.plot_test_latent_var, i_episode)
                            break
                        else:
                            line = ""
                            for ob in obs_unormalized2:
                                line += "," + str(ob)
                            line += "," + str(arg)
                            results.write(str(i_episode) + line + "\n")
                        action_results.write(
                            str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," + str(action[2]) + "\n")

                    print('{0},{1}'.format(i_episode, total_reward))
                    # exit(0)
                    discounted_rewards.extend(discount(r_i, 0.995))
                    total_rewards.append(total_reward)

                    # self.env.close()
                    # self.sess.close()
                return actions, obs, discounted_rewards, total_rewards
        else:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_D-Info_GAIL_results(Metar-3modes).csv'
            action_out_file = myconfig['output_dir'] + '/exp'+str(myconfig['exp'])+'_'+fname+'_D-Info_GAIL_actions_results(Metar-3modes).csv'

            enc_saver = tf.train.Saver(self.model2.weights)
            enc_saver.restore(self.sess, "VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")  # run2(batch-32-modes3)
            saver = tf.train.Saver(self.model.weights)
            saver.restore(self.sess, myconfig['output_dir']+'output/exp'+myconfig['exp']+"model.ckpt")

            with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
                results.write("episode,longitude,latitude,altitude,timestamp,Pressure_surface,"
                              "Relative_humidity_isobaric,Temperature_isobaric,Wind_speed_gust_surface,"
                              "u-component_of_wind_isobaric,v-component_of_wind_isobaric,drct,sknt,alti,vsby,gust,mode" +
                              "\n")
                action_results.write("episode,dlon,dlat,dalt\n")

                actions = []
                obs = []
                i = []
                discounted_rewards = []
                total_rewards = []
                print('Episodes,Reward')
                self.init = True

                for i_episode in range(episode_num):
                    self.counter2 = 0
                    self.plot_test_latent_var = []
                    self.plot_i_test = []
                    steps = 0
                    r_i = []
                    norm_observation, observation = self.env.reset()
                    obs_unormalized = unnormalize_observation_metar(observation)
                    #print('obs:', observation)
                    dec_input1, latent_init, arg = self.init_encoder(observation)

                    line = ""
                    for ob in obs_unormalized:
                        line += "," + str(ob)
                    line += "," + str(arg)
                    results.write(str(i_episode) + line + "\n")

                    total_reward = 0

                    for t in range(1000):
                        if self.init:
                            #print('latent:', latent_init)
                            #print('dec_input:', dec_input1)
                            obs.append(dec_input1)
                            action = self.act_test(dec_input1)
                            action = action.tolist()
                        else:
                            obs.append(dec_input2)
                            action = self.act_test(dec_input2)
                            action = action.tolist()

                        self.counter2 += 1

                        norm_observation2, observation2, reward, done = self.env.step(action, t)
                        # print('obs2:',observation2)

                        if self.init:
                            dec_input2, latent, arg = self.new_encoder(observation2, latent_init)
                            # print('latent:',latent)
                            # print('dec_in2:',dec_input2)
                        else:
                            dec_input2, latent, arg = self.new_encoder(observation2, latent)
                            # print('latent_2:',latent)
                            # print('dec_in2_2:',dec_input2)

                        steps += 1
                        r_i.append(reward)
                        actions.append(action)
                        total_reward += reward
                        action = unnormalize_action(action)
                        obs_unormalized2 = unnormalize_observation_metar(observation2)

                        self.plot_i_test.append(t)
                        self.init = False
                        if t % 100 == 0:
                            print("%i/%i" % (t + 100, 1000))
                        if t >= 1000 or done:
                            # if done:
                            #    print('')
                            # exit(0)
                            # continue

                            np1 = np.asarray(self.plot_i_test)
                            np2 = np.asarray(self.plot_test_latent_var)
                            print('np1_s:', np1.size)
                            print('np2_s:', np2.size)
                            self.plot_vae_test(self.plot_i_test, self.plot_test_latent_var, i_episode, '/VAE/results_gumbel_softmax/plot/aviation/epoch{}_aviation_latents.png')
                            break
                        else:
                            line = ""
                            for ob in obs_unormalized2:
                                line += "," + str(ob)
                            line += "," + str(arg)
                            results.write(str(i_episode) + line + "\n")
                        action_results.write(
                            str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," + str(action[2]) + "\n")

                    print('{0},{1}'.format(i_episode, total_reward))
                    # exit(0)
                    discounted_rewards.extend(discount(r_i, 0.995))
                    total_rewards.append(total_reward)

                    # self.env.close()
                    # self.sess.close()
                return actions, obs, discounted_rewards, total_rewards

    def rollout(self, mini_batch, latent_sequence_np):
        not_enough_samples = True
        batch_actions = []
        batch_observations = []
        batch_observations_lat = []
        batch_total_env_rewards = []
        log_observations = []
        log_actions = []
        episode = 0
        samples = 0
        global index
        global index2
        global counter_idx
        global indexing_csv

        if mini_batch == 0:
            indexing_csv = pd.read_csv('../../../aviation-indexing.csv')
            indexing_csv = np.asarray(indexing_csv).tolist()
            #indexing_csv = indexing_csv.tolist()

            index = 0
            index2 = indexing_csv[0][0]
            #print('index2:', index2)
            counter_idx = 0

        while not_enough_samples:
            episode += 1
            self.total_episodes += 1
            actions = []
            observations = []
            observations_lat = []
            raw_observation, observation = self.env.reset()
            total_env_reward = 0
            #print('index:', index, 'index2:', index2)
            latent_sequence = latent_sequence_np[index:index2]
            #print(latent_sequence)
            seq_end = len(latent_sequence)
            #print('seq_len:', seq_end)

            for t in range(self.path_size):
                #if (t>0):
                #    print('self.counter:', self.counter, 'done:', done, 'seq_size:', seq_end)
                action, _ = self.act(observation, latent_sequence)
                action = action.tolist()
                observation_lat = obs_matrix.tolist()
                observations.append(observation)
                observations_lat.append(observation_lat)

                if mini_batch % 100 == 0:
                    #log_observation = copy.deepcopy(raw_observation)
                    #log_observation.append(episode)
                    log_observation = observation_lat + [episode]
                    log_observations.append(log_observation)
                    log_action = copy.deepcopy(action)
                    log_action = np.append(log_action, episode)
                    log_actions.append(log_action)

                raw_observation, observation, reward_env, done = self.env.step(action, t)

                total_env_reward += reward_env

                actions.append(action)

                if t % 100 == 0:
                    #print("%i/%i" % (t + 100, self.path_size))
                    continue
                if (self.counter == seq_end) or done:
                    # print('DONE2')
                    # print('timestep:', t)
                    self.counter = 0
                    counter_idx += 1
                    index = index2
                    index2 += indexing_csv[counter_idx][0]
                    if (index2 >= 327071):
                        index = 0
                        index2 = indexing_csv[0][0]
                        counter_idx = 0
                    break

            samples += len(actions)
            batch_observations.append(observations)
            batch_observations_lat.append(observations_lat)
            batch_actions.append(actions)
            batch_total_env_rewards.append(total_env_reward)
            if samples >= self.mini_batch_size:
                not_enough_samples = False
                self.counter = 0

        if mini_batch % 100 == 0:
            np.savetxt(myconfig['exp']+'_'+str(mini_batch)+"_observation_log.csv", np.asarray(log_observations)
                       , delimiter=',', header='longitude,latitude,altitude,timestamp,'+
                                               'Pressure_surface,Relative_humidity_isobaric,'+
                                               'Temperature_isobaric,Wind_speed_gust_surface,'+
                                               'u-component_of_wind_isobaric,'+
                                               'v-component_of_wind_isobaric,'+
                                               'drct,sknt,alti,vsby,gust,'+'latent1,latent2,latent3,episode'
                       , comments='')
            np.savetxt(myconfig['exp']+'_'+str(mini_batch) + "_action_log.csv", np.asarray(log_actions),
                       delimiter=',', header='dlon,dlat,dalt,episode', comments='')

        return batch_observations_lat, batch_observations, batch_actions, batch_total_env_rewards

    def run_clonos(self, observation, init_latent):
        print('Running.. Encoder Clone')
        global global_concat_test1
        state = 0
        latent_counter = 1
        latent_flag = True
        init_latent = np.asarray(init_latent)
        while state < 327072:
            #print('state:', state)
            #if (latent_counter % 1000 == 0 or latent_counter == 1):
            if (latent_counter == 1):
                latent_flag = True
            else:
                latent_flag = False
                # print('counter:', counter, 'flag:', latent_flag)

            if (latent_flag == True):
                enc_input = np.concatenate((np.asarray(observation[state]), init_latent))
            else:
                enc_input = np.concatenate((np.asarray(observation[state]), global_concat_test1))

            latent_prob = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
            self.latent_sequence_prob.append(latent_prob)
            latent_prob = np.asarray(latent_prob)
            latent_new, argmax = self.one_hot_encoding(latent_prob)
            self.latent_sequence1.append(latent_new)
            sequence_prob = self.latent_sequence_prob
            sequence = self.latent_sequence1
            #print(latent_new)
            #print(sequence_prob[state])
            #print(sequence[state])
            # print(sequence)
            global_concat_test1 = latent_new
            state += 1
            latent_counter += 1
        self.latent_sequence_list = copy.deepcopy(self.latent_sequence1)
        np_arr_prob = np.asarray(self.latent_sequence_prob)
        np_arr = np.asarray(self.latent_sequence1)
        print(np_arr_prob.shape)
        print(np_arr.shape)
        print('Encoder Clone finished, mode sequences were created successfully!')
        #latent_sequence_prob_pd = pd.DataFrame(self.latent_sequence_prob, columns=['latent1', 'latent2', 'latent3'])
        #latent_sequence_pd = pd.DataFrame(self.latent_sequence1, columns=['latent1', 'latent2', 'latent3'])
        #latent_sequence_prob_pd.to_csv('./expert_data/latent_sequence_prob.csv', index=False)
        #latent_sequence_pd.to_csv('./expert_data/latent_sequence.csv', index=False)

        return np_arr_prob, np_arr

    def train(self, expert_observations, expert_actions):
        """
        Trains the agent.
        :return: void
        """

        encoder_saver = tf.train.Saver(self.model2.weights)
        encoder_saver.restore(self.sess,"./VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")
        saver = tf.train.Saver(self.model.weights)
        # saver.restore(self.sess, "./checkpoint/bcloning.ckpt")
        saver.restore(self.sess, "./VAE/results_gumbel_softmax/checkpoint/run7-enriched_Metar(3modes)/trpo_plugins/decoder/decoder_model_e2000.ckpt")
        discriminator_saver = tf.train.Saver(self.discriminator.discriminate.weights)

        latent_sequence_prob_np, latent_sequence_np = self.run_clonos(expert_observations, init_latent=[1., 0., 0.])
        latent_sequence_np = latent_sequence_np.tolist()
        latent_sequence_prob_np = latent_sequence_prob_np.tolist()

        print('Batches,Episodes,Surrogate,Reward,Env Reward')
        for mini_batch in range(self.mini_batches+1):

            # expert_observations_batch, expert_actions_batch = self.replay_buffer.get_batch(self.mini_batch_size)
            expert_observations_batch = expert_observations
            expert_actions_batch = expert_actions

            batch_observations_lat, batch_observations, batch_actions, batch_total_env_rewards = self.rollout(mini_batch, latent_sequence_np)

            flat_actions = [a for actions in batch_actions for a in actions]
            flat_observations = [o for observations in batch_observations_lat for o in observations]

            flat_actions = np.asarray(flat_actions, dtype=np.float64)
            flat_observations = np.asarray(flat_observations, dtype=np.float64)
            print('len:', len(flat_observations))
            flat_observations2 = flat_observations[:, :15]#4, 10, 15

            if mini_batch < self.mini_batches:
                d_loss_before_train, discriminator_loss = self.discriminator.train(expert_observations_batch, expert_actions_batch,
                                         flat_observations2[:self.mini_batch_size],
                                         flat_actions[:self.mini_batch_size])
            else:
                print('discriminator train not')

            batch_total_rewards = []
            batch_discounted_rewards_to_go = []
            batch_advantages = []
            total_reward = 0
            global d
            d = 0
            counters = 0
            index_rew = 0
            index_rew2 = 0
            for (observations, actions, obs_lat) in zip(batch_observations, batch_actions, batch_observations_lat):
                counters += len(observations)
                rewards_q, argmax_q = self.predict(np.asarray(obs_lat))
                # print('len:', len(rewards_q))
                # argmax_q = [np.argmax(i) for i in rewards_q]
                rewards_q = np.asarray(rewards_q)
                # print('rewards_q: \n', rewards_q)
                # print('argmax:', argmax_q)
                # print('len_argmax:', len(argmax_q))

                argmax_q2 = []
                for reward_q, i in zip(rewards_q, argmax_q):
                    element = reward_q[i]
                    argmax_q2.append(element)

                # print('argmax2:', argmax_q2)
                # print('len_argmax2:', len(argmax_q2))

                rewards_log = [np.log(i) for i in argmax_q2]
                # print('rewards_log:', rewards_log)
                # print('len_rew_log:', len(rewards_log))

                rewards_q2 = np.asarray([[i * 0.01] for i in rewards_log])
                #print('rewards_q:', rewards_q2)
                # print('len_rew_q:', len(rewards_q2))

                reward_t = -self.discriminator.predict(np.array(observations), np.array(actions))
                #print('rewards_t:', reward_t)
                # print('len_rew_t:', len(reward_t))
                # reward_t = [[i+t for i,t in zip(y, e)] for (y, e) in zip(reward_t, rewards_q2)]
                # print('reward_t_len:', len(reward_t), 'reward_q_len:', len(rewards_q2))
                # reward_t = reward_t + rewards_q2
                # print('reward_t_len:', len(reward_t))


                reward_t = np.asarray([i + j for i, j in zip(reward_t, rewards_q2)])
                #reward_t = [(sum(i,j)).tolist() for i,j in zip(reward_t, rewards_q2)]

                #print('rewards_t-after:', reward_t, '\n, type:', type(reward_t))
                #print('rewards_t-after:', reward_t)

                total_reward += np.sum(reward_t)
                batch_total_rewards.append(total_reward)
                reward_t = (reward_t.flatten())
                #print('flatten:', reward_t)
                discount_r = discount(reward_t, self.gamma)
                #print('discount:', discount_r)
                batch_discounted_rewards_to_go.extend(discount_r)
                obs_episode_np = np.array(observations)
                v = np.array(self.critic.predict(obs_episode_np)).flatten()
                v_next = shift(v, -1, cval=0)
                undiscounted_advantages = reward_t + self.gamma * v_next - v
                #print('undiscount:', undiscounted_advantages)

                discounted_advantages = discount(undiscounted_advantages, self.gamma * self.lamda)

                batch_advantages.extend(discounted_advantages)

            discounted_rewards_to_go_np = np.array(batch_discounted_rewards_to_go)
            discounted_rewards_to_go_np.resize((self.mini_batch_size, 1))

            observations_np = np.array(flat_observations2, dtype=np.float64) #10
            observations_np2 = np.array(flat_observations, dtype=np.float64) #13
            observations_np.resize((self.mini_batch_size, self.observation_dimensions))
            observations_np2.resize((self.mini_batch_size, self.observation_dimensions + self.latent_dimensions))

            advantages_np = np.array(batch_advantages)
            advantages_np.resize((self.mini_batch_size,))

            actions_np = np.array(flat_actions, dtype=np.float64).flatten()
            actions_np.resize((self.mini_batch_size, self.action_dimensions))

            self.critic.train(observations_np, discounted_rewards_to_go_np)
            feed = {self.x: observations_np2,
                    self.actions_ph: actions_np,
                    self.advantages_ph: advantages_np,
                    self.old_log_prob_ph: self.sess.run([self.log_prob], feed_dict={self.x: observations_np2, self.actions_ph: actions_np})
                    }

            g = np.array(self.sess.run([self.flat_grad],feed_dict=feed)[0],dtype=np.float64)
            step_dir = conjugate_gradient(self.__fisher_vector_product, g, feed)
            fvp = self.__fisher_vector_product(step_dir, feed)
            shs = step_dir.dot(fvp)
            assert shs > 0
            fullstep = np.sqrt(2 * self.max_kl / shs) * step_dir

            def loss_f(theta, mu_old):
                """
                Computes surrogate and KL of weights theta, used in
                line search.
                :param theta: Weights.
                :param mu_old: Distribution of old weights.
                :return: Vector [surrogate,KL]
                """
                feed[self.theta_ph] = theta
                feed[self.mu_old_ph] = mu_old
                self.sess.run([self.sff], feed_dict=feed)
                return self.sess.run(self.losses, feed_dict=feed)

            surrogate_run = self.sess.run(self.surrogate, feed_dict=feed)

            mu_old_run = self.sess.run(self.logits, feed_dict={self.x: observations_np2})
            theta_run = np.array(self.sess.run([self.flat_vars], feed_dict={})[0], dtype=np.float64)

            theta_new, surrogate_run = line_search(loss_f, theta_run,
                                                   fullstep, mu_old_run,
                                                   g.dot(step_dir),
                                                   surrogate_run,
                                                   self.max_kl)

            feed[self.theta_ph] = theta_new
            feed[self.step_direction] = step_dir
            _ = self.sess.run([self.sff], feed_dict=feed)
            if mini_batch % 10 == 0:
                _, summary = self.sess.run([self.step_direction, self.merged], feed_dict=feed)
                self.train_writer.add_summary(summary, mini_batch)

            self.episode_history.append(np.mean(batch_total_env_rewards))
            # mean = np.mean(self.episode_history)
            # if mean > max_mean:
            #     max_mean = mean
            #     saver.save(self.sess, myconfig['output_dir']+"output/exp"+str(myconfig['exp'])+"model.ckpt")

            print('{0},{1},{2},{3},{4},{5},{6}'.format(mini_batch, self.total_episodes,
                                           surrogate_run,np.mean(batch_total_rewards)
                                           ,np.mean(batch_total_env_rewards), d_loss_before_train, discriminator_loss))

            if mini_batch % 100 == 0:
                # encoder_saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "encoder_model.ckpt", global_step=mini_batch)
                saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt", global_step=mini_batch)
                discriminator_saver.save(self.discriminator.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "discriminator.ckpt", global_step=mini_batch)

        # encoder_saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "encoder_model.ckpt")
        saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt")
        discriminator_saver.save(self.discriminator.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "discriminator.ckpt")
