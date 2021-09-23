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
        df = pd.read_csv(myconfig['input_dir'] +
                         starting_points_file)[['longitude', 'latitude',
                                                'altitude','timestamp',
                                                'Pressure_surface', 'Relative_humidity_isobaric',
                                                'Temperature_isobaric', 'Wind_speed_gust_surface',
                                                'u-component_of_wind_isobaric',
                                                'v-component_of_wind_isobaric']]
        self.starting_points = df.values.tolist()
        self.s_point_index = 0
        self.random_choice = random_choice
        weather_df = pd.read_csv(myconfig['input_dir']+'dataset/final_exported_grib.csv')

        self.shape = []
        for idx, col in enumerate(['longitude','latitude','altitude','timestamp']):
            self.shape.append(weather_df[col].nunique())
        self.shape.append(6)
        print(self.shape)
        self.min_timestamp = weather_df['timestamp'].min()
        print(self.min_timestamp)
        self.min_lon = weather_df['longitude'].min()
        self.min_lat = weather_df['latitude'].min()

        self.weather_np = np.reshape(weather_df.drop(['longitude','latitude','altitude','timestamp','isobaric_level'], axis=1).to_numpy(), self.shape)
        self.iso_index = pd.read_csv(myconfig['input_dir']+'dataset/isobaric_index.csv').to_numpy()
        with open(myconfig['output_dir']+'output/exp'+str(myconfig['exp'])+'_env_log.csv', "w") as env_log:
            env_log.write('longitude,latitude,altitude,timestamp\n')

    def get_weather(self, lon, lat, alt, t):
        t_idx = int(((t + 10799) - self.min_timestamp) / 21600)
        lon_idx = int(((lon + 0.25) - self.min_lon) / 0.5)
        lat_idx = int(((lat + 0.25) - self.min_lat) / 0.5)
        idx = (np.abs(self.iso_index[:, 2] - alt)).argmin()
        alt_idx = 25 - int(self.iso_index[idx][0])

        if lon_idx >= self.shape[0] or lat_idx >= self.shape[1] or alt_idx >= self.shape[2]\
            or t_idx >= self.shape[3]:
            print('weather')
            self.done = True
            with open(myconfig['output_dir']+'output/exp'+str(myconfig['exp'])+'_env_log.csv', "a") as env_log:
                env_log.write(str(lon)+","+str(lat)+","+str(alt)+","+str(t)+"\n")
            return [0]*6

        return self.weather_np[lon_idx, lat_idx, alt_idx, t_idx]

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
                           (self.state[4] - myconfig['Pressure_surface_avg']) / myconfig['Pressure_surface_std'],
                           (self.state[5] - myconfig['Relative_humidity_isobaric_avg']) / myconfig['Relative_humidity_isobaric_std'],
                           (self.state[6] - myconfig['Temperature_isobaric_avg']) / myconfig['Temperature_isobaric_std'],
                           (self.state[7] - myconfig['Wind_speed_gust_surface_avg']) / myconfig['Wind_speed_gust_surface_std'],
                           (self.state[8] - myconfig['u-component_of_wind_isobaric_avg']) / myconfig['u-component_of_wind_isobaric_std'],
                           (self.state[9] - myconfig['v-component_of_wind_isobaric_avg']) / myconfig['v-component_of_wind_isobaric_std'],
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
                                                             'timestamp',
                                                             'Pressure_surface',
                                                             'Relative_humidity_isobaric',
                                                             'Temperature_isobaric',
                                                             'Wind_speed_gust_surface',
                                                             'u-component_of_wind_isobaric',
                                                             'v-component_of_wind_isobaric']]
        self.starting_points = df.values.tolist()

    def step(self, action, tstep):
        dlon = action[0]*myconfig['dlon_std'] + myconfig['dlon_avg']
        dlat = action[1]*myconfig['dlat_std'] + myconfig['dlat_avg']
        dalt = action[2]*myconfig['dalt_std'] + myconfig['dalt_avg']
        point = [self.state[0]+dlon, self.state[1]+dlat]
        alt = self.state[2]+dalt
        timestamp = self.state[3]+5
        self.state = [point[0], point[1], alt, timestamp]
        weather_vars = self.get_weather(point[0], point[1], alt, timestamp)
        self.state.extend(weather_vars)

        self.norm_state = [(point[0]-myconfig['longitude_avg'])/myconfig['longitude_std'],
                           (point[1]-myconfig['latitude_avg'])/myconfig['latitude_std'],
                           (alt-myconfig['altitude_avg'])/myconfig['altitude_std'],
                           (timestamp - myconfig['timestamp_avg']) / myconfig['timestamp_std'],
                           (self.state[4] - myconfig['Pressure_surface_avg']) / myconfig[
                               'Pressure_surface_std'],
                           (self.state[5] - myconfig['Relative_humidity_isobaric_avg']) / myconfig[
                               'Relative_humidity_isobaric_std'],
                           (self.state[6] - myconfig['Temperature_isobaric_avg']) / myconfig[
                               'Temperature_isobaric_std'],
                           (self.state[7] - myconfig['Wind_speed_gust_surface_avg']) / myconfig[
                               'Wind_speed_gust_surface_std'],
                           (self.state[8] - myconfig['u-component_of_wind_isobaric_avg']) /
                           myconfig['u-component_of_wind_isobaric_std'],
                           (self.state[9] - myconfig['v-component_of_wind_isobaric_avg']) /
                           myconfig['v-component_of_wind_isobaric_std'],
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

