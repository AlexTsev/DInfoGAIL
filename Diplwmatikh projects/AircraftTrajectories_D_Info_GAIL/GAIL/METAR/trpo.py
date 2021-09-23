# from myconfig import myconfig
from myconfig import myconfig
import tensorflow as tf
import copy
from tensorflow.python.keras.layers import Input, Dense, concatenate, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mean_squared_error
import numpy as np
from scipy.ndimage.interpolation import shift
from collections import deque
# import warnings
from critic import Critic
from utils import conjugate_gradient, set_from_flat, kl, self_kl, \
    flat_gradient, get_flat, discount, line_search, gauss_log_prob, visualize, gradient_summary, \
    unnormalize_action, unnormalize_observation_metar
# from  continuous.gail_atm.critic import Critic
# from continuous.gail_atm.utils import *

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
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5


class Policy(object):

    @staticmethod
    def create_policy(observation_dimensions, action_dimensions):
        """
        Creates the model of the policy.
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        :return: Model and the Input layer.
        """
        with tf.name_scope('Policy'):
            x = Input(shape=(observation_dimensions,), dtype=tf.float64)

            h = Dense(100, activation='tanh')(x)
            h1 = Dense(100, activation='tanh')(h)

            out = Dense(action_dimensions)(h1)

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
            disc_train_log.write("epoch,total_loss,expert_loss,policy_loss\n")


    def get_trainable_weights(self):
        return self.sess.run(
                [self.discriminate.trainable_weights], feed_dict={})[0]

    def train(self, expert_samples_observations, expert_samples_actions,
              policy_samples_observations, policy_samples_actions):
        with open(myconfig['output_dir']+'/disc_train_loss_log.csv', 'a+') as disc_train_log:
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

                disc_train_log.write(str(i)+","+str(loss_run)+","+str(expert_loss_run)+","+
                                     str(policy_loss_run)+"\n")
                # print('Discriminator loss:', loss_run)
                # if i % 100 == 0: print(i, "loss:", loss_run)
        return loss_before_train, loss_run

    def predict(self, samples_observations, samples_actions):
        return self.sess.run(self.log_D,
                             feed_dict={self.observations_input: samples_observations,
                                        self.actions_input: samples_actions})




class TRPOAgent(object):

    def __init__(self, env, observation_dimensions=4, action_dimensions=2):
        """
        Initializes the agent's parameters and constructs the flowgraph.
        :param env: Environment
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        """
        self.env = env
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.path_size = myconfig['path_size']
        self.mini_batch_size = myconfig['mini_batch_size']
        self.mini_batches = myconfig['mini_batches']
        self.gamma = myconfig['gamma']
        self.lamda = myconfig['lamda']
        self.max_kl = myconfig['max_kl']
        self.total_episodes = 0
        self.logstd = np.float64(myconfig['logstd'])
        self.critic = Critic(observation_dimensions=self.observation_dimensions)
        self.discriminator = Discriminator(observation_dimensions=self.observation_dimensions,
                                           action_dimensions=self.action_dimensions)
        # self.replay_buffer = ReplayBuffer()
        self.sess = tf.Session(config=config)
        self.model, self.x, self.h, self.h1 = Policy.create_policy(
            self.observation_dimensions, self.action_dimensions)
        visualize(self.model.trainable_weights)

        self.episode_history = deque(maxlen=100)

        self.advantages_ph = tf.placeholder(tf.float64, shape=None)
        self.actions_ph = tf.placeholder(tf.float64,
                                         shape=(None,
                                                action_dimensions),
                                         )
        self.old_log_prob_ph = tf.placeholder(tf.float64, shape=None)
        self.theta_ph = tf.placeholder(tf.float64, shape=None)
        self.tangent_ph = tf.placeholder(tf.float64, shape=None)
        self.mu_old_ph = tf.placeholder(tf.float64,
                                        shape=(None, action_dimensions))

        self.logits = self.model.outputs[0]

        var_list = self.model.trainable_weights
        self.flat_vars = get_flat(var_list)
        self.sff = set_from_flat(self.theta_ph, var_list)

        self.step_direction = tf.placeholder(tf.float64, shape=None)
        self.g_sum = gradient_summary(self.step_direction, var_list)

        # Compute surrogate.
        self.log_prob = gauss_log_prob(self.logits, self.logstd,
                                       self.actions_ph)
        neg_lh_divided = tf.exp(self.log_prob - self.old_log_prob_ph)
        w_neg_lh = neg_lh_divided * self.advantages_ph
        self.surrogate = tf.reduce_mean(w_neg_lh)

        kl_op = kl(self.logits, self.logstd, self.mu_old_ph,
                   self.logstd)
        self.losses = [self.surrogate, kl_op]

        self.flat_grad = flat_gradient(self.surrogate, var_list)
        # Compute fisher vector product
        self_kl_op = self_kl(self.logits, self.logstd)
        self_kl_flat_grad = flat_gradient(self_kl_op, var_list)
        g_vector_dotproduct = tf.reduce_sum(self_kl_flat_grad
                                            * self.tangent_ph)
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

    def act(self, observation):
        mu = self.sess.run(self.logits, feed_dict={self.x: [observation]})[0]
        act = mu + self.logstd * np.random.randn(self.action_dimensions)

        return act, mu

    def run(self, episode_num, bcloning=False, fname='0%_validate'):
        if bcloning:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_bcloning_results.csv'
            action_out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_actions_bcloning_results.csv'
        else:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_GAIL_results.csv'
            action_out_file = myconfig['output_dir'] + '/exp'+str(myconfig['exp'])+'_'+fname+'_GAIL_actions_results.csv'

        saver = tf.train.Saver(self.model.weights)
        if bcloning:
            saver.restore(self.sess,
                          myconfig['output_dir'] + "output/bcloning/bcloning.ckpt")
        else:
            saver.restore(self.sess, myconfig['output_dir'] + 'output/exp' + myconfig['exp'] + "model.ckpt")

        with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
            results.write("episode,longitude,latitude,altitude,timestamp,Pressure_surface,"
                          "Relative_humidity_isobaric,Temperature_isobaric,Wind_speed_gust_surface,"
                          "u-component_of_wind_isobaric,v-component_of_wind_isobaric,drct,sknt,alti,vsby,gust" +
                          # ",t_step,predalt" +
                          "\n")
            action_results.write("episode,dlon,dlat,dalt\n")
            actions = []
            obs = []
            discounted_rewards = []
            total_rewards = []
            print('Episodes, Reward')
            for i_episode in range(episode_num):
                r_i = []
                _, observation = self.env.reset()
                obs_unormalized = unnormalize_observation_metar(observation)
                line = ""
                for ob in obs_unormalized:
                    line += ","+str(ob)
                results.write(str(i_episode) + line +
                              "\n")
                # action_results.write(
                #     str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," +
                #     str(action[2]) + "\n")
                total_reward = 0

                for t in range(self.path_size):
                    obs.append(observation)
                    # action = self.act(observation)[0]
                    action = self.sess.run(self.logits, feed_dict={self.x: [observation]})[0]
                    _, observation, reward, done = self.env.step(action, t)
                    r_i.append(reward)
                    actions.append(action)
                    total_reward += reward
                    action = unnormalize_action(action)
                    obs_unormalized = unnormalize_observation_metar(observation)
                    if not done :
                        line = ""
                        for ob in obs_unormalized:
                            line += "," + str(ob)
                        results.write(str(i_episode) + line +
                                  "\n")
                    action_results.write(str(i_episode)+","+str(action[0])+","+str(action[1])+"," +
                                         str(action[2])+"\n")

                    if done:
                        break
                print('{0},{1}'.format(i_episode, total_reward))
                discounted_rewards.extend(discount(r_i, self.gamma))
                total_rewards.append(total_reward)
        return actions, obs, discounted_rewards, total_rewards

    def rollout(self, mini_batch):
        not_enough_samples = True
        batch_actions = []
        batch_observations = []
        batch_total_env_rewards = []
        log_observations = []
        log_actions = []
        episode = 0
        samples = 0

        while not_enough_samples:
            episode += 1
            self.total_episodes += 1
            actions = []
            observations = []
            raw_observation, observation = self.env.reset()
            total_env_reward = 0
            for t in range(self.path_size):
                observations.append(observation)
                action = self.act(observation)[0].tolist()
                if mini_batch % 100 == 0:
                    log_observation = copy.deepcopy(raw_observation)
                    log_observation.append(episode)
                    log_observations.append(log_observation)
                    log_action = copy.deepcopy(action)
                    log_action = np.append(log_action, episode)
                    log_actions.append(log_action)

                raw_observation, observation, reward_env, done = self.env.step(action, t)

                total_env_reward += reward_env

                actions.append(action)

                if done:
                    break

            samples += len(actions)
            batch_observations.append(observations)
            batch_actions.append(actions)
            batch_total_env_rewards.append(total_env_reward)
            if samples >= self.mini_batch_size:
                not_enough_samples = False
        if mini_batch % 100 == 0:
            np.savetxt(myconfig['exp']+'_'+str(mini_batch)+"_observation_log.csv", np.asarray(log_observations)
                       , delimiter=',', header='longitude,latitude,altitude,timestamp,'+
                                               'Pressure_surface,Relative_humidity_isobaric,'+
                                               'Temperature_isobaric,Wind_speed_gust_surface,'+
                                               'u-component_of_wind_isobaric,'+
                                               'v-component_of_wind_isobaric,'+'drct,'+'sknt,'+'alti,'+'vsby,'+'gust,' +
                                               # + ',t_step,predalt,
                                               'episode'
                       , comments='')
            np.savetxt(myconfig['exp']+'_'+str(mini_batch) + "_action_log.csv", np.asarray(log_actions),
                       delimiter=',', header='dlon,dlat,dalt,episode', comments='')

        return batch_observations, batch_actions, batch_total_env_rewards

    def train(self, expert_observations, expert_actions):
        """
        Trains the agent.
        :return: void
        """

        # self.replay_buffer.seed_buffer(expert_observations, expert_actions)
        saver = tf.train.Saver(self.model.weights)
        saver.restore(self.sess, myconfig['output_dir'] + "output/bcloning/bcloning.ckpt")
        discriminator_saver = tf.train.Saver(self.discriminator.discriminate.weights)
        print('Batches,Episodes,Surrogate,Reward,Env Reward')
        for mini_batch in range(self.mini_batches+1):

            # expert_observations_batch, expert_actions_batch = self.replay_buffer.get_batch(self.mini_batch_size)
            expert_observations_batch = expert_observations
            expert_actions_batch = expert_actions

            batch_observations, batch_actions, batch_total_env_rewards = self.rollout(mini_batch)
            flat_actions = [a for actions in batch_actions for a in actions]
            flat_observations = [o for observations in batch_observations for o in observations]

            if mini_batch < self.mini_batches:
                d_loss_before_train, discriminator_loss = self.discriminator.train(expert_observations_batch, expert_actions_batch,
                                         flat_observations[:self.mini_batch_size],
                                         flat_actions[:self.mini_batch_size])
            else:
                print('discriminator train not')

            batch_total_rewards = []
            batch_discounted_rewards_to_go = []
            batch_advantages = []
            total_reward = 0
            for (observations, actions) in zip(batch_observations, batch_actions):
                reward_t = -self.discriminator.predict(np.array(observations), np.array(actions))

                total_reward += np.sum(reward_t)
                batch_total_rewards.append(total_reward)
                reward_t = (reward_t.flatten()).tolist()

                batch_discounted_rewards_to_go.extend(discount(reward_t, self.gamma))
                obs_episode_np = np.array(observations)
                v = np.array(self.critic.predict(obs_episode_np)).flatten()
                v_next = shift(v, -1, cval=0)
                undiscounted_advantages = reward_t + self.gamma * v_next - v

                discounted_advantages = discount(undiscounted_advantages,
                                                 self.gamma * self.lamda)

                batch_advantages.extend(discounted_advantages)

            discounted_rewards_to_go_np = np.array(batch_discounted_rewards_to_go)
            discounted_rewards_to_go_np.resize((self.mini_batch_size, 1))

            observations_np = np.array(flat_observations, dtype=np.float64)
            observations_np.resize((self.mini_batch_size, self.observation_dimensions))

            advantages_np = np.array(batch_advantages)
            advantages_np.resize((self.mini_batch_size,))

            actions_np = np.array(flat_actions,
                                  dtype=np.float64).flatten()
            actions_np.resize((self.mini_batch_size, self.action_dimensions))

            self.critic.train(observations_np, discounted_rewards_to_go_np)
            feed = {self.x: observations_np,
                    self.actions_ph: actions_np,
                    self.advantages_ph: advantages_np,
                    self.old_log_prob_ph: self.sess.run(
                        [self.log_prob], feed_dict={
                            self.x: observations_np,
                            self.actions_ph: actions_np})
                    }

            g = np.array(self.sess.run([self.flat_grad],
                                       feed_dict=feed)[0],
                         dtype=np.float64
                         )
            step_dir = conjugate_gradient(self.__fisher_vector_product,
                                          g, feed)
            fvp = self.__fisher_vector_product(
                step_dir, feed)
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

            surrogate_run = self.sess.run(self.surrogate,
                                          feed_dict=feed)

            mu_old_run = self.sess.run(self.logits,
                                       feed_dict={
                                           self.x: observations_np
                                       })
            theta_run = np.array(self.sess.run(
                [self.flat_vars], feed_dict={})[0], dtype=np.float64)

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
                saver.save(self.sess,
                           myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt",
                           global_step=mini_batch)
                discriminator_saver.save(self.discriminator.sess,
                                         myconfig['output_dir'] + "output/exp" + str(
                                             myconfig['exp']) + "discriminator.ckpt",
                                         global_step=mini_batch)

        saver.save(self.sess,
                   myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt")
        discriminator_saver.save(self.discriminator.sess,
                                 myconfig['output_dir'] + "output/exp" + str(
                                     myconfig['exp']) + "discriminator.ckpt")
