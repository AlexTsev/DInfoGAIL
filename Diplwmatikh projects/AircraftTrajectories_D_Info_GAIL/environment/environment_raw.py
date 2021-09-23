from geo.sphere import distance, destination
#from geopy import distance,destination
import numpy as np
import copy
import random
import pandas as pd
from myconfig import myconfig

conf = {
        'mbr_top_left': (-3.7038, 41.4),
        'mbr_bot_right': (2.9504, 39.9864),
        'td': 5,
        'state': [2.0635461669776305,41.27632140995424,940.0],
        'origin': (2.0833, 41.2974),  # Barcelona
        'destination': (-3.5676, 40.4983),  # Madrid
        'destination_distance': 5000,
        'done': False,
        'max_alt': 40000}


class Environment(object):

    def __init__(self, random_choice=True, fname='train'):
        self.mbr_top_left = conf['mbr_top_left']
        self.mbr_bot_right = conf['mbr_bot_right']
        self.td = conf['td']
        self.state = conf['state']
        self.norm_state = []
        self.destination = conf['destination']
        self.done = conf['done']
        self.max_alt = conf['max_alt']
        self.t_step = 0
        starting_points_file = 'dataset/'+fname+'_starting_points.csv'
        # if validate:
        #     starting_points_file = '/validate_starting_points.csv'
        # else:
        #     starting_points_file = '/train_starting_points.csv'
        df = pd.read_csv(myconfig['input_dir'] + starting_points_file)[['longitude', 'latitude','altitude','timestamp']]
        self.starting_points = df.values.tolist()
        self.s_point_index = 0
        self.random_choice = random_choice

        with open(myconfig['output_dir']+'output/exp'+str(myconfig['exp'])+'_env_log.csv', "w") as env_log:
            env_log.write('longitude,latitude,altitude,timestamp\n')


    def next_starting_point(self):
        s_point = self.starting_points[self.s_point_index]
        self.s_point_index = (self.s_point_index+1) % len(self.starting_points)

        return s_point

    def reset(self):
        self.mbr_top_left = conf['mbr_top_left']
        self.mbr_bot_right = conf['mbr_bot_right']
        self.td = conf['td']
        if not self.random_choice:
            self.state = copy.deepcopy(self.next_starting_point())
        else:
            self.state = copy.deepcopy(random.choice(self.starting_points))

        self.norm_state = [(self.state[0]-myconfig['longitude_avg'])/myconfig['longitude_std'],
                           (self.state[1]-myconfig['latitude_avg'])/myconfig['latitude_std'],
                           (self.state[2]-myconfig['altitude_avg'])/myconfig['altitude_std'],
                           (self.state[3] - myconfig['timestamp_avg']) / myconfig['timestamp_std'],
                           ]

        self.destination = conf['destination']
        self.done = conf['done']
        self.max_alt = conf['max_alt']

        return self.state, self.norm_state

    def read_starting_points(self, random_choice, fname):
        self.random_choice = random_choice
        self.s_point_index = 0
        starting_points_file = '/'+fname+'_starting_points.csv'
        # if validate:
        #     starting_points_file = '/validate_starting_points.csv'
        # else:
        #     starting_points_file = '/train_starting_points.csv'
        df = pd.read_csv(myconfig['input_dir'] + starting_points_file)[
                                                            ['longitude',
                                                             'latitude',
                                                             'altitude',
                                                             'timestamp']]
        self.starting_points = df.values.tolist()

    def step(self, action, tstep):
        dlon = action[0]*myconfig['dlon_std'] + myconfig['dlon_avg']
        dlat = action[1]*myconfig['dlat_std'] + myconfig['dlat_avg']
        dalt = action[2]*myconfig['dalt_std'] + myconfig['dalt_avg']
        point = [self.state[0]+dlon, self.state[1]+dlat]
        alt = self.state[2]+dalt
        timestamp = self.state[3]+5
        self.state = [point[0], point[1], alt, timestamp]

        self.norm_state = [(point[0]-myconfig['longitude_avg'])/myconfig['longitude_std'],
                           (point[1]-myconfig['latitude_avg'])/myconfig['latitude_std'],
                           (alt-myconfig['altitude_avg'])/myconfig['altitude_std'],
                           (timestamp - myconfig['timestamp_avg']) / myconfig['timestamp_std'],
                           ]
        reward = 0
        if distance(point, self.destination) < conf['destination_distance']:
            reward = 0
            self.done = True
        elif point[0] < self.mbr_top_left[0] \
                or point[1] > self.mbr_top_left[1]\
                or point[0] > self.mbr_bot_right[0]\
                or point[1] < self.mbr_bot_right[1]\
                or alt > self.max_alt\
                or alt < 0\
                or tstep == myconfig['path_size']-1:

            reward = -distance(point, self.destination)*myconfig['env_reward_lambda']
            #print('reward:',reward)
            self.done = True

        return self.state, self.norm_state, reward, self.done

    def validate(self, point):
        self.done = False
        if distance(point, self.destination) < conf['destination_distance']\
                or point[0] < self.mbr_top_left[0]\
                or point[1] > self.mbr_top_left[1]\
                or point[0] > self.mbr_bot_right[0]\
                or point[1] < self.mbr_bot_right[1]\
                or self.state[2] > self.max_alt:

            self.done = True

        return self.done

