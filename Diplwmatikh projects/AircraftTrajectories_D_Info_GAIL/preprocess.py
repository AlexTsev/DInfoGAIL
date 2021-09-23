#!/usr/bin/python
from myconfig import myconfig
import trpo
import critic
<<<<<<< HEAD
from environment import environment_metar as environment
=======
from environment import environment_raw as environment
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from statistics import mean
import pandas as pd
from tensorflow.python.keras.utils import plot_model
from sklearn.model_selection import KFold
import tensorflow as tf
# from continuous.gail_atm.myconfig import myconfig
# from continuous.gail_atm import trpo
# from  continuous.gail_atm import critic
# from  continuous.gail_atm.environment import environment
<<<<<<< HEAD
=======
from tensorflow.python.client import device_lib
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

config = tf.ConfigProto()
<<<<<<< HEAD
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
=======
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

def readArgs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--experiment', default="0")
    parser.add_argument('-s', '--split_dataset', nargs='?', const=True, default=False)
    parser.add_argument('-l', '--log_dir', default="./")
    parser.add_argument('-i', '--input_dir', default="./")
    parser.add_argument('-o', '--output_dir', default="./")

    args = parser.parse_args()
    return args


def bcloning(x_train, y_train, x_test, y_test):
    model = agent.model
    x = agent.x

    sess = tf.Session(config=config)
    checkpoint_path = myconfig['output_dir']+"output/bcloning/bcloning.ckpt"
    # discriminator_path = myconfig['output_dir']+"/bcloning_discriminator.ckpt"
    predictions = model.outputs[0]
    labels = tf.placeholder(tf.float64, shape=(None), name='y')
    loss = tf.reduce_mean(tf.square(predictions - labels))
    opt = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(model.weights)
    # discriminator_saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    # saver.restore(sess, checkpoint_path)

    epochs = myconfig['bcloning_epochs']
    batch_size = 64
    kf = KFold(n_splits=myconfig['bcloning_folds'], shuffle=True)
    print(kf.get_n_splits(x_train))
    # step = 0

    #print(x_train[0], y_train[0], x_test[0], y_test[0])
    #x_train, y_train = shuffle(x_train, y_train, random_state=12)
    #x_test, y_test = shuffle(x_test, y_test, random_state=12)
    #print('shuffle:')
    #print(x_train[0], y_train[0], x_test[0], y_test[0])

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
                if len(x_t[j:j+batch_size]) < 64:
                    continue
                _, loss_run, _ = sess.run([opt, loss, labels],
                                                           feed_dict={x: x_t[j:j + batch_size],
                                                                      labels: y_t[j:j + batch_size]})
                train_loss.append(loss_run)

            val_loss_run, _ = sess.run([loss, labels],
                                          feed_dict={x: x_val,
                                                     labels: y_val})
            print('epoch',i,'train loss',mean(train_loss),'validation_loss',val_loss_run)

    saver.save(sess, checkpoint_path)
    test_loss_run, _ = sess.run([loss, labels],
                               feed_dict={x: x_test,
                                          labels: y_test})

    print(test_loss_run)


def collect_samples(dir, agent, episode_num):

    actions, observations, discounted_rewards, total_rewards = agent.run(episode_num)
    np.savez_compressed(dir + 'demo'+str(episode_num)+'.npz', actions=actions,
                        observations=observations,
                        discounted_rewards=discounted_rewards,
                        total_rewards=total_rewards)


def subsample(obs_train, actions_train):

    subsampled_observations = []
    subsampled_actions = []
    assert len(obs_train) == len(actions_train)

    for idx, sample in enumerate(zip(obs_train, actions_train)):
        if idx % 14 == 0:
            subsampled_observations.append(sample[0])
            subsampled_actions.append(sample[1])

    return subsampled_observations, subsampled_actions



def load_samples(dir, file):
    demo = np.load(dir + file)

    return demo


def plot_policy():
    model, input = trpo.Policy.create_policy(action_dimensions=3,
<<<<<<< HEAD
                                             observation_dimensions=3)
=======
                                       observation_dimensions=3)
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
    plot_model(model, show_shapes=True, to_file='policy_model.png')


def plot_critic():
    critic_model = critic.Critic(observation_dimensions=3)

    plot_model(critic_model.model, show_shapes=True, to_file='critic_model.png')

def plot_discriminator():
    discriminator = trpo.Discriminator(action_dimensions=3,
                                       observation_dimensions=3)
    plot_model(discriminator.discriminator, show_shapes=True, to_file='discriminator_model.png')
    plot_model(discriminator.discriminate, show_shapes=True, to_file='discriminate_model.png')


def test_discriminator(observations_test_expert, actions_test_expert, observations_test_generator,
                       actions_test_generator,gen_file,expert_file):

    saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    saver.restore(agent.discriminator.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "discriminator.ckpt")
    # expert_pred = np.exp(agent.discriminator.predict(observations_test_expert,actions_test_expert))
    expert_reward = -agent.discriminator.predict(observations_test_expert,actions_test_expert)
    # generator_pred = np.exp(agent.discriminator.predict(observations_test_generator,actions_test_generator))
    generator_reward = -agent.discriminator.predict(observations_test_generator,actions_test_generator)
    np.savetxt(expert_file,expert_reward,delimiter=",",header='expert_reward',comments='')
    np.savetxt(gen_file,generator_reward,delimiter=",",header='generator_reward',comments='')
    print("Expert avg:", np.mean(np.exp(-expert_reward)))
    print("Expert std:", np.std(np.exp(-expert_reward)))
    print("Expert min:", np.min(np.exp(-expert_reward)))
    print("Expert max:", np.max(np.exp(-expert_reward)))
    print("Generator avg:", np.mean(np.exp(-generator_reward)))
    print("Generator std:", np.std(np.exp(-generator_reward)))
    print("Generator min:",np.min(np.exp(-generator_reward)))
    print("Generator max:",np.max(np.exp(-generator_reward)))


def plot_discriminator_predict():
    sns.set(rc={'figure.figsize':(18,10)})
    expert = pd.read_csv("expert_train_reward_"+myconfig['exp']+".csv")
    expert_test = pd.read_csv("expert_test_reward_"+myconfig['exp']+".csv")
    generator = pd.read_csv("generator_train_reward_"+myconfig['exp']+".csv")
    generator_test = pd.read_csv("generator_test_reward_"+myconfig['exp']+".csv")

    df = pd.concat([expert,expert_test,generator,generator_test],axis=1)
    df.columns = ['expert_reward','expert_test_reward','generator_reward','generator_test_reward']
    df_pow = np.exp(-df)

    sns.distplot(df_pow['expert_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'expert_train_set'})
    sns.distplot(df_pow['expert_test_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'expert_test_set'})
    sns.distplot(df_pow['generator_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'generator_train_set'})
    sns.distplot(df_pow['generator_test_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'generator_test_set'})
    plt.savefig(myconfig['plot_dir']+'/discriminator_all.png')
    plt.clf()
    sns.distplot(df_pow['expert_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'expert_train_set'},color='cyan')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_trainset.png')
    plt.clf()
    sns.distplot(df_pow['expert_test_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'expert_test_set'},color='darkorange')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_testset.png')
    plt.clf()
    sns.distplot(df_pow['generator_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'generator_train_set'},color='green')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_traintset.png')
    plt.clf()
    sns.distplot(df_pow['generator_test_reward'].dropna(),axlabel='discriminator prediction',kde_kws={'label':'generator_test_set'},color='red')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_testset.png')



def discriminator_load_test():

    for files in [[myconfig['output_dir'] + "/exp" + str(myconfig['exp']) + "_0%_test_GAIL_results.csv",
                  myconfig['output_dir'] + "/exp" + str(myconfig['exp']) + "_0%_test_GAIL_actions_results.csv",
                   'generator_test_reward_' + myconfig['exp'] + '.csv',
                   'expert_test_reward_' + myconfig['exp'] + '.csv', obs_test, actions_test],
                  [myconfig['exp'] + "_0_observation_log.csv",
                   myconfig['exp'] + "_0_action_log.csv",
                   'generator_train_reward_' + myconfig['exp'] + '.csv',
                   'expert_train_reward_' + myconfig['exp'] + '.csv', obs_train, actions_train]]:
        # 'generator_test_reward.csv','expert_test_reward.csv',obs_test,actions_test],
        #          ["0_0_observation_log.csv","0_0_action_log.csv",'generator_train_reward.csv',
        #           'expert_train_reward.csv',obs_train,actions_train]]:

        generator_observations = pd.read_csv(files[0]).drop(['episode'], axis=1)
        generator_actions = pd.read_csv(files[1]).drop(['episode'], axis=1)

        generator_actions['dlon'] = (generator_actions['dlon']-myconfig['dlon_avg'])/myconfig['dlon_std']
        generator_actions["dlat"] = (generator_actions['dlat'] -
                                              myconfig['dlat_avg']) / \
                                              myconfig['dlat_std']
        generator_actions["dalt"] = (generator_actions['dalt'] - myconfig['dalt_avg']) / \
                                      myconfig['dalt_std']

        generator_observations['longitude'] = (generator_observations['longitude']-myconfig['longitude_avg'])/myconfig['longitude_std']
        generator_observations['latitude'] = (generator_observations['latitude'] - myconfig['latitude_avg']) / myconfig['latitude_std']
        generator_observations['altitude'] = (generator_observations['altitude'] - myconfig['altitude_avg']) / myconfig[
                'altitude_std']
        generator_observations['timestamp'] = (generator_observations['timestamp'] - myconfig['timestamp_avg']) / myconfig[
                'timestamp_std']
<<<<<<< HEAD
        generator_observations['Pressure_surface'] = (generator_observations['Pressure_surface'] - myconfig['Pressure_surface_avg']) / myconfig[
            'Pressure_surface_std']
        generator_observations['Relative_humidity_isobaric'] = (generator_observations['Relative_humidity_isobaric'] - myconfig['Relative_humidity_isobaric_avg']) / myconfig[
            'Relative_humidity_isobaric_std']
        generator_observations['Temperature_isobaric'] = (generator_observations['Temperature_isobaric'] - myconfig['Temperature_isobaric_avg']) / myconfig[
            'Temperature_isobaric_std']
        generator_observations['Wind_speed_gust_surface'] = (generator_observations['Wind_speed_gust_surface'] - myconfig['Wind_speed_gust_surface_avg']) / myconfig[
            'Wind_speed_gust_surface_std']
        generator_observations['u-component_of_wind_isobaric'] = (generator_observations['u-component_of_wind_isobaric'] - myconfig['u-component_of_wind_isobaric_avg']) / myconfig[
            'u-component_of_wind_isobaric_std']
        generator_observations['v-component_of_wind_isobaric'] = (generator_observations['v-component_of_wind_isobaric'] - myconfig['v-component_of_wind_isobaric_avg']) / myconfig[
            'v-component_of_wind_isobaric_std']
=======
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

        generator_actions = generator_actions.values
        generator_observations = generator_observations.values

        test_discriminator(files[4], files[5], generator_observations, generator_actions, files[2], files[3])

    plot_discriminator_predict()


def conclude_actions(file):
    expert = pd.read_csv(file)
    next_states = expert[['longitude','latitude','altitude']][1:].reset_index(drop=True)
    states = expert[['longitude', 'latitude', 'altitude']][:-1].reset_index(drop=True)
    action_trajectory_ID = expert['trajectory_ID'][1:].reset_index(drop=True)
    expert_new = expert[:-1].reset_index(drop=True)
    expert_actions = next_states.subtract(states,axis=1)
    expert_new['dlon'] = expert_actions['longitude']
    expert_new['dlat'] = expert_actions['latitude']
    expert_new['dalt'] = expert_actions['altitude']
    expert_new['action_trajectoryID'] = action_trajectory_ID
    expert_new = expert_new[expert_new['action_trajectoryID'] == expert_new['trajectory_ID']]
    expert_new = expert_new.drop(['action_trajectoryID'],axis=1)
    expert_new.to_csv(myconfig['input_dir']+'dataset/enriched_expert_hierarchical_clustering_ds.csv',index=False)

<<<<<<< HEAD

=======
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
def dataset_cluster_split(file):
    df = pd.read_csv(file)
    c1 = df[df['Cluster'] == 1]
    c1_flights = c1['trajectory_ID'].sort_values().unique()
    c1_flights = shuffle(c1_flights)
    c1_test_flights = c1_flights[:25]
    c1_validation_flights = c1_flights[25:50]
    c1_train_flights = c1_flights[50:]
    c1_train_validation_flights = c1_flights[25:]
    c2 = df[df['Cluster'] == 2]
    c2_flights = c2['trajectory_ID'].sort_values().unique()
    c2_flights = shuffle(c2_flights)
    c2_test_flights = c2_flights[:25]
    c2_validation_flights = c2_flights[25:50]
    c2_train_flights = c2_flights[50:]
    c2_train_validation_flights = c2_flights[25:]

    c1_test_set = df[df['trajectory_ID'].isin(c1_test_flights)]
    c1_validation_set = df[df['trajectory_ID'].isin(c1_validation_flights)]
    c1_train_set = df[df['trajectory_ID'].isin(c1_train_flights)]
    c1_train_validation_set = df[df['trajectory_ID'].isin(c1_train_validation_flights)]

    c1_test_set.to_csv(myconfig['input_dir']+'dataset/c1_test_set.csv',index=False,header=True)
    c1_validation_set.to_csv(myconfig['input_dir'] + 'dataset/c1_validation_set.csv', index=False,
                             header=True)
    c1_train_set.to_csv(myconfig['input_dir'] + 'dataset/c1_train_set.csv', index=False,
                             header=True)
    c1_train_validation_set.to_csv(myconfig['input_dir'] + 'dataset/c1_train_validation_set.csv',
                                   index=False, header=True)

    c2_test_set = df[df['trajectory_ID'].isin(c2_test_flights)]
    c2_validation_set = df[df['trajectory_ID'].isin(c2_validation_flights)]
    c2_train_set = df[df['trajectory_ID'].isin(c2_train_flights)]
    c2_train_validation_set = df[df['trajectory_ID'].isin(c2_train_validation_flights)]

    c2_test_set.to_csv(myconfig['input_dir']+'dataset/c2_test_set.csv',index=False,header=True)
    c2_validation_set.to_csv(myconfig['input_dir'] + 'dataset/c2_validation_set.csv', index=False,
                             header=True)
    c2_train_set.to_csv(myconfig['input_dir'] + 'dataset/c2_train_set.csv', index=False,
                             header=True)
    c2_train_validation_set.to_csv(myconfig['input_dir'] + 'dataset/c2_train_validation_set.csv',
                                   index=False, header=True)
    test_set_df = c1_test_set.append(c2_test_set)
    test_set_df.to_csv(myconfig['input_dir']+'dataset/test_set.csv',index=False,
                                           header=True)
    c1_validation_set.append(c2_validation_set).to_csv(myconfig['input_dir']+'dataset/validation_set.csv',
                                                       index=False,header=True)
    c1_train_set.append(c2_train_set).to_csv(myconfig['input_dir']+'dataset/train_set.csv',index=False,
                                             header=True)
    train_validation_set_df = c1_train_validation_set.append(c2_train_validation_set)
    train_validation_set_df.to_csv(myconfig['input_dir']+
                                                                   '/train_validation_set.csv',
                                                                   index=False, header=True)

    return train_validation_set_df, test_set_df

<<<<<<< HEAD

=======
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
def split_dataset(file,splits):
    df = pd.read_csv(file)
    flights = df['trajectory_ID'].sort_values().unique()
    flights = shuffle(flights)
    flight_num = df['trajectory_ID'].sort_values().nunique()
    split_flight_num = int(flight_num / splits)
    train_idx = split_flight_num*(splits-2)
    print('Train flights',train_idx)
    train_flights = flights[:train_idx]
    validation_idx = train_idx+split_flight_num
    print('Validation flights', validation_idx-train_idx )
    validation_flights = flights[train_idx:validation_idx]
    test_flights = flights[validation_idx:]
    print('Validation flights', flights.size - validation_idx)

    obs_train = df[df['trajectory_ID'].isin(train_flights)].drop(['dlon','dlat',
                                                                  'great_circle_distance',
                                                                  'temporal_distance','dalt',
                                                                  'isobaric_level'],
                                                                 axis=1)
    obs_validate = df[df['trajectory_ID'].isin(validation_flights)].drop(['dlon','dlat',
                                                                  'great_circle_distance',
                                                                  'temporal_distance','dalt',
                                                                  'isobaric_level'],
                                                                 axis=1)
    obs_test = df[df['trajectory_ID'].isin(test_flights)].drop(
        ['dlon', 'dlat',
         'great_circle_distance',
         'temporal_distance', 'dalt',
         'isobaric_level'],
        axis=1)
    print(obs_train.head())
    actions_train = df[df['trajectory_ID'].isin(train_flights)][['trajectory_ID','dlon','dlat','dalt']]
    actions_validate = df[df['trajectory_ID'].isin(validation_flights)][['trajectory_ID','dlon','dlat','dalt']]
    actions_test = df[df['trajectory_ID'].isin(test_flights)][['trajectory_ID','dlon','dlat','dalt']]
    print(actions_train.head())

    obs_train.to_csv(myconfig['input_dir']+'dataset/obs_train.csv',index=False,header=True)
    obs_validate.to_csv(myconfig['input_dir']+'dataset/obs_validate.csv', index=False, header=True)
    obs_test.to_csv(myconfig['input_dir']+'dataset/obs_test.csv', index=False, header=True)
    actions_train.to_csv(myconfig['input_dir']+'dataset/actions_train.csv', index=False, header=True)
    actions_validate.to_csv(myconfig['input_dir']+'dataset/actions_validate.csv', index=False, header=True)
    actions_test.to_csv(myconfig['input_dir']+'dataset/actions_test.csv', index=False, header=True)

    return obs_train, actions_train,obs_validate,actions_validate, obs_test, actions_test


def test_weather():
    # expert_np = pd.read_csv('dataset/enriched_expert.csv').drop([
    #     'trajectory_ID','isobaric_level','dlon','dlat','great_circle_distance','temporal_distance','dalt'
    #     ], axis=1).to_numpy()

    expert_np = pd.read_csv('dataset/raw/test_LEBL_LEMD_01_24_raw.csv').drop([
        'trajectory_ID'
    ], axis=1).to_numpy()

    for row in expert_np:

        # weather = env.get_weather(row[0],row[1],row[2],row[3])
        if row[1] < -3.7038:
            continue
        weather = env.get_weather(row[1], row[2], row[3], row[0])
        if 1-all(weather):

            continue

        for (r,w) in zip(row[-6:],weather[-6:]):

            if round(r,5) != round(w,5):
                print('error')
                print(row[0])
                print(r,w)
                print(row)
                print(weather)
                exit(0)

def validate_weather():
    expert_np = pd.read_csv('dataset/enriched_expert_clear.csv').drop([
        'trajectory_ID','isobaric_level','bearing','constant_speed','great_circle_distance','temporal_distance','v_speed'
        ], axis=1).to_numpy()

    for row in expert_np:
        weather = env.get_weather(row[0],row[1],row[2],row[3])
        for (r,w) in zip(row[-6:],weather):
            if round(r,5) != round(w,5):
                print('error')
                print(row[3])
                print(r,w)
                print(row)
                print(weather)
                exit(0)


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

def normalize_observations_raw(obs):
    obs['longitude'] = (obs['longitude']-myconfig['longitude_avg'])/myconfig['longitude_std']
    obs['latitude'] = (obs['latitude'] - myconfig['latitude_avg']) / myconfig['latitude_std']
    obs['altitude'] = (obs['altitude'] - myconfig['altitude_avg']) / myconfig['altitude_std']
    obs['timestamp'] = (obs['timestamp'] - myconfig['timestamp_avg']) / myconfig['timestamp_std']

    return obs

def normalize_observations_metar(obs):
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

    obs['drct'] = (obs['drct'] - myconfig['drct_avg']) / myconfig['drct_std']
    obs['sknt'] = (obs['sknt'] - myconfig['sknt_avg']) / myconfig['sknt_std']
    obs['alti'] = (obs['alti'] - myconfig['alti_avg']) / myconfig['alti_std']
    obs['vsby'] = (obs['vsby'] - myconfig['vsby_avg']) / myconfig['vsby_std']
    obs['gust'] = (obs['gust'] - myconfig['gust_avg']) / myconfig['gust_std']

    return obs


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
<<<<<<< HEAD
                                                     'altitude', 'timestamp', 'Pressure_surface',
                                                     'Relative_humidity_isobaric',
                                                     'Temperature_isobaric',
                                                     'Wind_speed_gust_surface',
                                                     'u-component_of_wind_isobaric',
                                                     'v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']).to_csv(
=======
                                                     'altitude', 'timestamp']).to_csv(
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
            myconfig['input_dir'] + '/' + str(int(n[i]*100)) +
            '%_' + fname + '_starting_points.csv',
            index=False,
        )
        pd.DataFrame(trajectories_prcnt[i], columns=['trajectory_ID', 'longitude', 'latitude',
<<<<<<< HEAD
                                                     'altitude', 'timestamp', 'Pressure_surface',
                                                     'Relative_humidity_isobaric',
                                                     'Temperature_isobaric',
                                                     'Wind_speed_gust_surface',
                                                     'u-component_of_wind_isobaric',
                                                     'v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']).to_csv(
=======
                                                     'altitude', 'timestamp']).to_csv(
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
            myconfig['input_dir'] + '/' + str(int(n[i]*100)) +
            '%_' + fname + '_trajectories.csv',
            index=False,
        )


def normalize_actions(actions):
    actions['dlon'] = (actions['dlon']-myconfig['dlon_avg'])/myconfig['dlon_std']
    actions["dlat"] = (actions['dlat'] -
                                       myconfig['dlat_avg']) / \
                                       myconfig['dlat_std']
    actions["dalt"] = (actions['dalt'] - myconfig['dalt_avg']) / \
                                    myconfig['dalt_std']

    return actions


def trajectoryID_results(fname):
    generated_obs_df = pd.read_csv(myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_GAIL_results.csv')
    bcloning_obs_df = pd.read_csv(myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_bcloning_results.csv')
    tIDs = obs_test_df['trajectory_ID'].unique().tolist()
    episodes = generated_obs_df['episode'].unique().tolist()
    generated_obs_df['trajectory_ID'] = generated_obs_df['episode'].replace(episodes, tIDs)
    generated_obs_df = generated_obs_df.drop(columns=['episode'],axis=1)
    bcloning_obs_df['trajectory_ID'] = bcloning_obs_df['episode'].replace(episodes, tIDs)
    bcloning_obs_df = bcloning_obs_df.drop(columns=['episode'], axis=1)

    generated_obs_df.to_csv(myconfig['output_dir']+'/exp'+myconfig['exp']+'_'+fname+'_GAIL_results_tID.csv',
                     columns=['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp'],
                     index=False)
    bcloning_obs_df.to_csv(
        myconfig['output_dir'] + '/exp' + myconfig['exp'] + '_' + fname + '_bcloning_results_tID.csv',
        columns=['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp'],
        index=False)


episode_num = 25
args = readArgs()
myconfig['input_dir'] = args.input_dir
myconfig['output_dir'] = args.output_dir
myconfig['exp'] = args.experiment
myconfig['log_dir'] = args.log_dir
# myconfig['plot_dir'] = './plots/'+args.experiment

print("Reading demonstrations")
if args.split_dataset:
    print("split_dataset")
    conclude_actions(myconfig['input_dir']+'dataset/enriched_expert_hierarchical_clustering.csv')
    train_validation_set_df, test_set_df = dataset_cluster_split(myconfig['input_dir']+'dataset/enriched_expert_hierarchical_clustering_ds.csv')

    obs_train = train_validation_set_df[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                         'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
<<<<<<< HEAD
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']]
    obs_test = test_set_df[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                         'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']]
=======
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric']]
    obs_test = test_set_df[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                         'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric']]
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

    starting_points_2(obs_test, 'test')

    obs_train.groupby('trajectory_ID').\
        head(1).to_csv(myconfig['input_dir']+'dataset/train_starting_points.csv', index=False)


else:
    test_df = pd.read_csv(myconfig['input_dir']+'dataset/test_set.csv')

<<<<<<< HEAD
    obs_test_df = test_df[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                         'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']]
=======
    obs_test_df = test_df[['trajectory_ID','longitude','latitude','altitude','timestamp']]
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
    starting_points_2(obs_test_df, 'test')
    test_flight_num = obs_test_df['trajectory_ID'].nunique()
    obs_test = obs_test_df.drop(['trajectory_ID'], axis=1)
    actions_test = test_df[['dlon','dlat','dalt']]
    print('test')
    print(obs_test.head())
    print(actions_test.head())

    train_df = pd.read_csv(myconfig['input_dir'] + 'dataset/train_set.csv')

<<<<<<< HEAD
    obs_train_df = train_df[['trajectory_ID','longitude','latitude','altitude','timestamp','Pressure_surface',
                                         'Relative_humidity_isobaric','Temperature_isobaric','Wind_speed_gust_surface',
                                         'u-component_of_wind_isobaric','v-component_of_wind_isobaric','drct', 'sknt', 'alti', 'vsby', 'gust']]
=======
    obs_train_df = train_df[['trajectory_ID','longitude','latitude','altitude','timestamp']]
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

    obs_train_df.groupby('trajectory_ID').head(1).to_csv(myconfig['input_dir']+'dataset/train_starting_points.csv', index=False)

    train_flight_num = obs_train_df['trajectory_ID'].nunique()
    print('train_fl_num:', train_flight_num)

    obs_train = obs_train_df.drop(['trajectory_ID'],axis=1)
    actions_train = train_df[['dlon','dlat','dalt']]
    print('train')
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

<<<<<<< HEAD
    myconfig['Pressure_surface_avg'] = obs_train.loc[:, "Pressure_surface"].mean()
    myconfig['Pressure_surface_std'] = obs_train.loc[:, "Pressure_surface"].std()
    myconfig['Relative_humidity_isobaric_avg'] = obs_train.loc[:,"Relative_humidity_isobaric"].mean()
    myconfig['Relative_humidity_isobaric_std'] = obs_train.loc[:,"Relative_humidity_isobaric"].std()
    myconfig['Temperature_isobaric_avg'] = obs_train.loc[:, "Temperature_isobaric"].mean()
    myconfig['Temperature_isobaric_std'] = obs_train.loc[:, "Temperature_isobaric"].std()
    myconfig['Wind_speed_gust_surface_avg'] = obs_train.loc[:, "Wind_speed_gust_surface"].mean()
    myconfig['Wind_speed_gust_surface_std'] = obs_train.loc[:, "Wind_speed_gust_surface"].std()
    myconfig['u-component_of_wind_isobaric_avg'] = obs_train.loc[:, "u-component_of_wind_isobaric"].mean()
    myconfig['u-component_of_wind_isobaric_std'] = obs_train.loc[:, "u-component_of_wind_isobaric"].std()
    myconfig['v-component_of_wind_isobaric_avg'] = obs_train.loc[:, "v-component_of_wind_isobaric"].mean()
    myconfig['v-component_of_wind_isobaric_std'] = obs_train.loc[:, "v-component_of_wind_isobaric"].std()

    myconfig['drct_avg'] = obs_train.loc[:, "drct"].mean()
    myconfig['drct_std'] = obs_train.loc[:, "drct"].std()
    myconfig['sknt_avg'] = obs_train.loc[:, "sknt"].mean()
    myconfig['sknt_std'] = obs_train.loc[:, "sknt"].std()
    myconfig['alti_avg'] = obs_train.loc[:, "alti"].mean()
    myconfig['alti_std'] = obs_train.loc[:, "alti"].std()
    myconfig['vsby_avg'] = obs_train.loc[:, "vsby"].mean()
    myconfig['vsby_std'] = obs_train.loc[:, "vsby"].std()
    myconfig['gust_avg'] = obs_train.loc[:, "gust"].mean()
    myconfig['gust_std'] = obs_train.loc[:, "gust"].std()

    obs_train = normalize_observations_metar(obs_train)
    actions_train = normalize_actions(actions_train)
    # obs_validate = normalize_observations(obs_validate)
    # actions_validate = normalize_actions(actions_validate)
    obs_test = normalize_observations_metar(obs_test)
    actions_test = normalize_actions(actions_test)

    #actions_train.to_csv(myconfig['input_dir']+'dataset/actions_train_normalized.csv')
=======
    #myconfig['Pressure_surface_avg'] = obs_train.loc[:, "Pressure_surface"].mean()
    #myconfig['Pressure_surface_std'] = obs_train.loc[:, "Pressure_surface"].std()
    #myconfig['Relative_humidity_isobaric_avg'] = obs_train.loc[:,"Relative_humidity_isobaric"].mean()
    #myconfig['Relative_humidity_isobaric_std'] = obs_train.loc[:,"Relative_humidity_isobaric"].std()
    #myconfig['Temperature_isobaric_avg'] = obs_train.loc[:, "Temperature_isobaric"].mean()
    #myconfig['Temperature_isobaric_std'] = obs_train.loc[:, "Temperature_isobaric"].std()
    #myconfig['Wind_speed_gust_surface_avg'] = obs_train.loc[:, "Wind_speed_gust_surface"].mean()
    #myconfig['Wind_speed_gust_surface_std'] = obs_train.loc[:, "Wind_speed_gust_surface"].std()
    #myconfig['u-component_of_wind_isobaric_avg'] = obs_train.loc[:, "u-component_of_wind_isobaric"].mean()
    #myconfig['u-component_of_wind_isobaric_std'] = obs_train.loc[:, "u-component_of_wind_isobaric"].std()
    #myconfig['v-component_of_wind_isobaric_avg'] = obs_train.loc[:, "v-component_of_wind_isobaric"].mean()
    #myconfig['v-component_of_wind_isobaric_std'] = obs_train.loc[:, "v-component_of_wind_isobaric"].std()

    #myconfig['drct_avg'] = obs_train.loc[:, "drct"].mean()
    #myconfig['drct_std'] = obs_train.loc[:, "drct"].std()
    #myconfig['sknt_avg'] = obs_train.loc[:, "sknt"].mean()
    #myconfig['sknt_std'] = obs_train.loc[:, "sknt"].std()
    #myconfig['alti_avg'] = obs_train.loc[:, "alti"].mean()
    #myconfig['alti_std'] = obs_train.loc[:, "alti"].std()
    #myconfig['vsby_avg'] = obs_train.loc[:, "vsby"].mean()
    #myconfig['vsby_std'] = obs_train.loc[:, "vsby"].std()
    #myconfig['gust_avg'] = obs_train.loc[:, "gust"].mean()
    #myconfig['gust_std'] = obs_train.loc[:, "gust"].std()

    obs_train = normalize_observations_raw(obs_train)
    actions_train = normalize_actions(actions_train)
    # obs_validate = normalize_observations(obs_validate)
    # actions_validate = normalize_actions(actions_validate)
    obs_test = normalize_observations_raw(obs_test)
    actions_test = normalize_actions(actions_test)

    actions_train.to_csv(myconfig['input_dir']+'dataset/actions_train_normalized.csv')
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
    # actions_validate.to_csv(myconfig['input_dir'] + '/actions_validate_normalized.csv')
    actions_test.to_csv(myconfig['input_dir'] + 'dataset/actions_test_normalized.csv')

    print(obs_train.head())


    actions_train = actions_train.values
    obs_train = obs_train.values
    # obs_validate = obs_validate.values
    # actions_validate = actions_validate.values
    actions_test = actions_test.values
    obs_test = obs_test.values


env = environment.Environment(random_choice=True, fname='train')
<<<<<<< HEAD
agent = trpo.TRPOAgent(env, action_dimensions=3, latent_dimensions=5, observation_dimensions=15)
=======
agent = trpo.TRPOAgent(env, action_dimensions=3, latent_dimensions=5, observation_dimensions=4)
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89
print("Starting Training")
print('obs_tr:', len(obs_train), 'act_tr:', len(actions_train))


#bcloning(obs_train, actions_train, obs_test, actions_test)
#print('obs_tr:', len(obs_train), 'act_tr:', len(actions_train))
agent.train(obs_train, actions_train)
#for prcnt in [0,20,50,70]:
#fname = str('prcnt')+'%_test'
fname = '0%_test'

env.read_starting_points(random_choice=False, fname=fname)
actions_vae, obs_vae, discounted_rewards_vae, total_rewards_vae = agent.run(test_flight_num, vae=True, bcloning=False, fname=fname)

# env.read_starting_points(random_choice=True,fname='train')

# env.read_starting_points(random_choice=False, fname=fname)
actions_gail, obs_gail, discounted_rewards_gail, total_rewards_gail = agent.run(test_flight_num, vae=False, bcloning=False, fname=fname)

print('obs_train:', len(obs_train), 'actions_train:', len(actions_train), 'obs_test:', len(obs_test), 'actions_test:', len(actions_test))
#print('actions_BC:', len(actions_bcloning), 'obs_BC:', len(obs_bcloning))
print('actions_VAE:', len(actions_vae), 'obs_VAE:', len(obs_vae))
print('actions_gail:', len(actions_gail), 'obs_gail:', len(obs_gail), '\n')

#print('Sum of Rewards BC:', sum(total_rewards_bcloning))  # np.mean(total_rewards_vae)
print('Sum of Rewards VAE:', sum(total_rewards_vae))  # np.mean(total_rewards_vae)
<<<<<<< HEAD
print('Sum of Rewards Directed-Info Gail:', sum(total_rewards_gail))  # np.mean(total_rewards_gail)

#print('Mean Reward BC:', np.mean(total_rewards_bcloning))
print('Mean Reward VAE:', np.mean(total_rewards_vae))
print('Mean Reward Directed-Info Gail:', np.mean(total_rewards_gail))
=======
print('Sum of Rewards Gail:', sum(total_rewards_gail))  # np.mean(total_rewards_gail)

#print('Mean Reward BC:', np.mean(total_rewards_bcloning))
print('Mean Reward VAE:', np.mean(total_rewards_vae))
print('Mean Reward Gail:', np.mean(total_rewards_gail))
>>>>>>> eae2860dcbace7eab2353b7eb0af709d393c2f89

#trajectoryID_results(fname)


#discriminator_load_test()
