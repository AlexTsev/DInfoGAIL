from myconfig import myconfig
#from environment import H
#from environment import hopper
import gym
from statistics import mean
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import VAE.vae_gumbel_softmax as VAE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import VAE.hw1.tf_util as tf_util
from GAIL import trpo as trpo
from tensorflow.python.keras.utils import plot_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
not_trained = False

def bcloning(x_train, y_train, x_test, y_test):
    model = agent.model
    x = agent.x

    sess = tf.Session(config=config)
    checkpoint_path = "./checkpoint/bcloning.ckpt"
    # discriminator_path = myconfig['output_dir']+"output/bcloning_discriminator.ckpt"
    predictions = model.outputs[0]
    labels = tf.placeholder(tf.float64, shape=(None), name='y')
    loss = tf.reduce_mean(tf.square(predictions - labels))
    opt = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(model.weights)
    # discriminator_saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    # saver.restore(sess, checkpoint_path)

    epochs = myconfig['bcloning_epochs']
    batch_size=64
    kf = KFold(n_splits=myconfig['bcloning_folds'], shuffle=True)
    count = 0
    print(kf.get_n_splits(x_train))
    # step = 0
    for train_index, validation_index in kf.split(x_train):
        count += 1
        x_t = x_train[train_index]
        x_val = x_train[validation_index]
        y_t = y_train[train_index]
        y_val = y_train[validation_index]
        print('New Fold',count)
        for i in range(epochs):
            train_loss = []
            # preds = []
            for j in range(0, len(x_t), batch_size):
                if len(x_t[j:j+batch_size]) < 64:
                    continue
                _, loss_run, _ = sess.run([opt, loss, labels],
                                                           feed_dict={x: x_t[j:j + batch_size],
                                                                      labels: y_t[j:j + batch_size]})
                train_loss.append(loss_run)

            val_loss_run, _ = sess.run([loss, labels],
                                          feed_dict={x: x_val,
                                                     labels: y_val})
            print('epoch', i, 'train loss', mean(train_loss), 'validation_loss', val_loss_run)

    saver.save(sess, checkpoint_path)
    test_loss_run, _ = sess.run([loss, labels],
                               feed_dict={x: x_test,
                                          labels: y_test})

    print(test_loss_run)

def vae(x_train, y_train, x_test, y_test):
    if (not_trained):
        model = agent.model
        x = agent.x

        sess = tf.Session(config=config)
        checkpoint_path = "checkpoint/vae.ckpt"
        predictions = model.outputs[0]
        labels = tf.placeholder(tf.float64, shape=(None), name='y')
        #loss
        loss = tf.reduce_mean(tf.square(predictions - labels))
        opt = tf.train.AdamOptimizer().minimize(loss)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(model.weights)
        epochs = myconfig['vae_epochs']
        batch_size= 32
        kf = KFold(n_splits=myconfig['vae_folds'], shuffle=True)

        print(kf.get_n_splits(x_train))
        # step = 0
        for train_index, validation_index in kf.split(x_train):
            x_t = x_train[train_index]
            x_val = x_train[validation_index]
            y_t = y_train[train_index]
            y_val = y_train[validation_index]
            print('New Fold')
            for i in range(epochs):
                train_loss = []
                # preds = []
                for j in range(0, len(x_t), batch_size):
                    if len(x_t[j:j+batch_size]) < 32:
                        continue
                    _, loss_run, _ = sess.run([opt, loss, labels],
                                                               feed_dict={x: x_t[j:j + batch_size],
                                                                          labels: y_t[j:j + batch_size]})
                    train_loss.append(loss_run)

                val_loss_run, _ = sess.run([loss, labels],
                                              feed_dict={x: x_val,
                                                         labels: y_val})
                print('epoch', i, 'train loss', mean(train_loss), 'validation_loss', val_loss_run)

        saver.save(sess, checkpoint_path)
        test_loss_run, _ = sess.run([loss, labels],
                                   feed_dict={x: x_test,
                                              labels: y_test})

        print(test_loss_run)
    else:
        model = agent.model
        x = agent.x
        sess = tf.Session(config=config)
        saver = tf.train.Saver(model.weights)
        checkpoint_path = "checkpoint/vae.ckpt"
        saver.restore(sess, checkpoint_path)

#------------------------------------------------------------------------------------------------------#

def pandas1(obs):
    df_norm_obs = pd.DataFrame(obs, columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10','var11'])
    return df_norm_obs

def pandas2(actions):
    df_norm = pd.DataFrame(actions, columns=['action1', 'action2', 'action3'])
    return df_norm

def apply_normalization_observations(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm_obs = pd.DataFrame(np_scaled, columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11'])
    return df_norm_obs

def apply_normalization_actions(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm = pd.DataFrame(np_scaled, columns=['action1', 'action2', 'action3'])
    return df_norm
#obs = pandas1(obs)
#actions = pandas2(actions)

#Load the Dataset
print('Loading Dataset...')
obs = pd.read_csv('../dataset/hopper-v2/hopper_v2_observations(62000).csv')
actions = pd.read_csv('../dataset/hopper-v2/hopper_v2_actions(62000).csv')
print('len:',len(obs))

#x_train = obs[:50000]
#x_test = obs[50000:]
#y_train = actions[:50000]
#y_test = actions[50000:]

#x_train = apply_normalization_observations(x_train)
#x_test = apply_normalization_observations(x_test)
#print('obs_normalized:')
#print(obs_normalized)
#y_train = apply_normalization_actions(y_train)
#y_test = apply_normalization_actions(y_test)
#print('actions_normalized:')
#print(actions_normalized)


x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2, shuffle=True)

x_train = pandas1(x_train)
x_test = pandas1(x_test)
y_train = pandas2(y_train)
y_test = pandas2(y_test)

print(len(x_train),len(x_test), len(y_train), len(y_test))

def plot_discriminator():
    discriminator = trpo.Discriminator(action_dimensions=3, observation_dimensions=11)
    plot_model(discriminator.discriminator, show_shapes=True, to_file='discriminator_model.png')
    plot_model(discriminator.discriminate, show_shapes=True, to_file='discriminate_model.png')

def test_discriminator(observations_test_expert, actions_test_expert, observations_test_generator, actions_test_generator, gen_file, expert_file):
    saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    saver.restore(agent.discriminator.sess, "./.output/exp0discriminator.ckpt")
    # expert_pred = np.exp(agent.discriminator.predict(observations_test_expert,actions_test_expert))
    expert_reward = -agent.discriminator.predict(observations_test_expert, actions_test_expert)
    # generator_pred = np.exp(agent.discriminator.predict(observations_test_generator,actions_test_generator))
    generator_reward = -agent.discriminator.predict(observations_test_generator, actions_test_generator)
    np.savetxt(expert_file, expert_reward, delimiter=",", header='expert_reward', comments='')
    np.savetxt(gen_file, generator_reward, delimiter=",", header='generator_reward', comments='')
    print("Expert avg:", np.mean(np.exp(-expert_reward)))
    print("Expert std:", np.std(np.exp(-expert_reward)))
    print("Expert min:", np.min(np.exp(-expert_reward)))
    print("Expert max:", np.max(np.exp(-expert_reward)))
    print("Generator avg:", np.mean(np.exp(-generator_reward)))
    print("Generator std:", np.std(np.exp(-generator_reward)))
    print("Generator min:", np.min(np.exp(-generator_reward)))
    print("Generator max:", np.max(np.exp(-generator_reward)))

def plot_discriminator_predict():
    sns.set(rc={'figure.figsize':(18,10)})
    expert = pd.read_csv("expert_train_reward_"+str(myconfig['exp'])+".csv")
    expert_test = pd.read_csv("expert_test_reward_"+str(myconfig['exp'])+".csv")
    generator = pd.read_csv("generator_train_reward_"+str(myconfig['exp'])+".csv")
    generator_test = pd.read_csv("generator_test_reward_"+str(myconfig['exp'])+".csv")

    df = pd.concat([expert, expert_test, generator, generator_test],axis=1)
    df.columns = ['expert_reward', 'expert_test_reward', 'generator_reward', 'generator_test_reward']
    df_pow = np.exp(-df)

    sns.distplot(df_pow['expert_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_train_set'})
    sns.distplot(df_pow['expert_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_test_set'})
    sns.distplot(df_pow['generator_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_train_set'})
    sns.distplot(df_pow['generator_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_test_set'})
    plt.savefig(myconfig['plot_dir']+'/discriminator_all.png')
    plt.clf()
    sns.distplot(df_pow['expert_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_train_set'},color='cyan')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_trainset.png')
    plt.clf()
    sns.distplot(df_pow['expert_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_test_set'},color='darkorange')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_testset.png')
    plt.clf()
    sns.distplot(df_pow['generator_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_train_set'},color='green')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_traintset.png')
    plt.clf()
    sns.distplot(df_pow['generator_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_test_set'},color='red')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_testset.png')


def discriminator_load_test():
    for files in [['./exp0_0%_test_GAIL_results.csv',#obs,
                   './exp0_0%_test_GAIL_actions_results.csv',#actions,
                   'generator_test_reward_' + str(myconfig['exp']) + '.csv',
                   'expert_test_reward_' + str(myconfig['exp']) + '.csv', x_test, y_test],
                  [str(myconfig['exp']) + '_0_observation_log.csv',
                   str(myconfig['exp']) + '_0_action_log.csv',
                   'generator_train_reward_' + str(myconfig['exp']) + '.csv',
                   'expert_train_reward_' + str(myconfig['exp']) + '.csv', x_train, y_train]]:

        generator_observations = pd.read_csv(files[0]).drop(['episode'], axis=1)#files[0]
        generator_actions = pd.read_csv(files[1]).drop(['episode'], axis=1)#files[1]

        generator_actions['action1'] = (generator_actions['action1'] - myconfig['action1_avg']) / myconfig['action1_std']
        generator_actions['action2'] = (generator_actions['action2'] - myconfig['action2_avg']) / myconfig['action2_std']
        generator_actions['action3'] = (generator_actions['action3'] - myconfig['action3_avg']) / myconfig['action3_std']

        generator_observations['var1'] = (generator_observations['var1'] - myconfig['var1_avg']) / myconfig['var1_std']
        generator_observations['var2'] = (generator_observations['var2'] - myconfig['var2_avg']) / myconfig['var2_std']
        generator_observations['var3'] = (generator_observations['var3'] - myconfig['var3_avg']) / myconfig['var3_std']
        generator_observations['var4'] = (generator_observations['var4'] - myconfig['var4_avg']) / myconfig['var4_std']
        generator_observations['var5'] = (generator_observations['var5'] - myconfig['var5_avg']) / myconfig['var5_std']
        generator_observations['var6'] = (generator_observations['var6'] - myconfig['var6_avg']) / myconfig['var6_std']
        generator_observations['var7'] = (generator_observations['var7'] - myconfig['var7_avg']) / myconfig['var7_std']
        generator_observations['var8'] = (generator_observations['var8'] - myconfig['var8_avg']) / myconfig['var8_std']
        generator_observations['var9'] = (generator_observations['var9'] - myconfig['var9_avg']) / myconfig['var9_std']
        generator_observations['var10'] = (generator_observations['var10'] - myconfig['var10_avg']) / myconfig['var10_std']
        generator_observations['var11'] = (generator_observations['var11'] - myconfig['var11_avg']) / myconfig['var11_std']

        generator_actions = generator_actions.values
        generator_observations = generator_observations.values

        #print('gen_actions:', len(generator_actions))
        #print('gen_obs:', len(generator_observations))
        print('########################################################')
        test_discriminator(files[4], files[5], generator_observations, generator_actions, files[2], files[3])
    print('########################################################')
    plot_discriminator_predict()

myconfig['action1_avg'] = y_train.loc[:, "action1"].mean()
myconfig['action1_std'] = y_train.loc[:, "action1"].std()
myconfig['action2_avg'] = y_train.loc[:, "action2"].mean()
myconfig['action2_std'] = y_train.loc[:, "action2"].std()
myconfig['action3_avg'] = y_train.loc[:, "action3"].mean()
myconfig['action3_std'] = y_train.loc[:, "action3"].std()

myconfig['var1_avg'] = x_train.loc[:, "var1"].mean()
myconfig['var1_std'] = x_train.loc[:, "var1"].std()
myconfig['var2_avg'] = x_train.loc[:, "var2"].mean()
myconfig['var2_std'] = x_train.loc[:, "var2"].std()
myconfig['var3_avg'] = x_train.loc[:, "var3"].mean()
myconfig['var3_std'] = x_train.loc[:, "var3"].std()
myconfig['var4_avg'] = x_train.loc[:, "var4"].mean()
myconfig['var4_std'] = x_train.loc[:, "var4"].std()
myconfig['var5_avg'] = x_train.loc[:, "var5"].mean()
myconfig['var5_std'] = x_train.loc[:, "var5"].std()
myconfig['var6_avg'] = x_train.loc[:, "var6"].mean()
myconfig['var6_std'] = x_train.loc[:, "var6"].std()
myconfig['var7_avg'] = x_train.loc[:, "var7"].mean()
myconfig['var7_std'] = x_train.loc[:, "var7"].std()
myconfig['var8_avg'] = x_train.loc[:, "var8"].mean()
myconfig['var8_std'] = x_train.loc[:, "var8"].std()
myconfig['var9_avg'] = x_train.loc[:, "var9"].mean()
myconfig['var9_std'] = x_train.loc[:, "var9"].std()
myconfig['var10_avg'] = x_train.loc[:, "var10"].mean()
myconfig['var10_std'] = x_train.loc[:, "var10"].std()
myconfig['var11_avg'] = x_train.loc[:, "var11"].mean()
myconfig['var11_std'] = x_train.loc[:, "var11"].std()

#x_train = pandas1(x_train)
#x_test = pandas1(x_test)
#y_train = pandas2(y_train)
#y_test = pandas2(y_test)

actions_train = y_train.values
obs_train = x_train.values
actions_test = y_test.values
obs_test = x_test.values
# obs_validate = obs_validate.values
# actions_validate = actions_validate.values

env = gym.make("Hopper-v2")
#env.reset()
observation_space = env.observation_space
action_space = env.action_space
observation_space_Dim = observation_space.shape[0]
action_space_Dim = action_space.shape[0]

print(observation_space, observation_space_Dim)
print(action_space, action_space_Dim)

agent = trpo.TRPOAgent(env, action_dimensions=action_space_Dim, observation_dimensions=observation_space_Dim)
#agent = trpo.TRPOAgent(env, action_dimensions=action_space_Dim, latent_dimensions=3, observation_dimensions=observation_space_Dim)

print("Starting Training")
obs_train = np.asarray(obs_train)
actions_train = np.asarray(actions_train)
obs_test = np.asarray(obs_test)
actions_test = np.asarray(actions_test)

#IF BEHAVIORAL CLONING
#bcloning(obs_train, actions_train, obs_test, actions_test)
#IF VAE
#vae(obs_train, actions_train, obs_test, actions_test)

#Train Agent
agent.train(obs_train, actions_train)
fname = '0%_test'
#Test Agent
actions_bcloning, obs_bcloning, discounted_rewards_bcloning, total_rewards_bcloning = agent.run(50, bcloning=True, fname=fname)
actions_gail, obs_gail, discounted_rewards_gail, total_rewards_gail = agent.run(50, bcloning=False, fname=fname)

print('obs_train:', len(obs_train), 'actions_train:', len(actions_train), 'obs_test:', len(obs_test), 'actions_test:', len(actions_test))
print('actions_BC:', len(actions_bcloning), 'obs_BC:', len(obs_bcloning))
print('actions_gail:', len(actions_gail), 'obs_gail:', len(obs_gail), '\n')

print('Sum of Rewards BC:', sum(total_rewards_bcloning))#np.mean(total_rewards_vae)
print('Sum of Rewards Gail:', sum(total_rewards_gail))#np.mean(total_rewards_gail)

print('Mean Reward BC:', np.mean(total_rewards_bcloning))
print('Mean Reward Gail:', np.mean(total_rewards_gail))

#print(actions_gail)
#print(obs_gail)

#Load Discriminator
discriminator_load_test()

env.close()
