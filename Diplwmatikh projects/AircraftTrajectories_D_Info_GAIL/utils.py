import tensorflow as tf
import numpy as np
import scipy.signal
import random
from myconfig import myconfig


def conjugate_gradient(f_Ax, b, feed, cg_iters=10, residual_tol=1e-10):
    """
    Computes approximately (H**-1)g using the conjugate gradient method.
    :param f_Ax: Function to compute fisher vector product H*g.
    :param b: Vector g.
    :param feed: Dictionary, feed_dict for tf.placeholders.
    :param cg_iters: Total number of iterations.
    :param residual_tol: Upper bound of approximation.
    :return: Approximation of (H**-1)g.
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        # try:
        z = f_Ax(p, feed)
        v = rdotr / p.dot(z)
        x = x + v*p
        r = r - v*z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        assert z.shape == p.shape and p.shape == x.shape \
            and x.shape == r.shape, "Conjugate shape difference"
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
        # except Warning as warning:
        #     print(warning)
        #     exit(1)
    return x


def visualize(vars):
    for var in vars:
        tf.summary.histogram('weights/'+var.name, var)


def gradient_summary(flat_grad, var_list):
    start = 0
    for var in var_list:
        end = start+np.prod(var.shape)
        tf.summary.histogram('gradients/'+var.name, tf.reshape(flat_grad[start:end], var.shape))


def set_from_flat(theta_ph, var_list):
    """
    Creates an operation that sets the neural network's weights to the
    values of theta_ph (vector).
    :param theta_ph: New weights' vector.
    :param var_list: Neural network's weights.
    :return: Operation that sets the neural network's weights to the
    values of theta_ph (vector).
    """
    start = 0
    assigns = []
    for var in var_list:
        end = start+np.prod(var.shape)
        assigns.append(tf.assign(var, tf.reshape(theta_ph[start:end],
                                                 var.shape)))
        start = end

    group = tf.group(*assigns)
    return group


def kl(mu1, logstd1, mu2, logstd2):
    """
    Creates an operation that computes KL divergence between two
    distributions.
    :param mu1: Mean.
    :param logstd1: Log of standard deviation.
    :param mu2: Mean.
    :param logstd2: Log of standard deviation.
    :return: Operation that computes KL divergence between
    distributions.
    """
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)
    return tf.reduce_sum(
        logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (
                    2 * var2) - 0.5)


def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2 * var) - logstd \
         - .5*tf.log(tf.constant(2*np.pi, dtype=tf.float64))
    return tf.reduce_sum(gp, [1])


def self_kl(mu, logstd):
    """
    Creates operation that computes KL(p,p as const). We need this in
    order to compute gradient w.r.t. theta of KL(theta,theta_old).
    :param mu: Mean.
    :param logstd: Log of standard deviation.
    :return: Operation that computes
    KL(p(mu,logstd) || p(mu as const,logstd as const)).
    """
    mu1 = tf.stop_gradient(mu)
    logstd1 = tf.stop_gradient(logstd)
    return kl(mu1, logstd1, mu, logstd)


def flat_gradient(f, var_list):
    """
    Creates operation that computes and flattens the gradient of a
    function w.r.t. the weights.
    :param f: Function of which we compute the gradient.
    :param var_list: List of the weights.
    :return: Operation that computes and flattens the gradient of a
    function w.r.t. the weights.
    """
    grads = tf.gradients(f, var_list)
    flatgrad = tf.concat([tf.reshape(grad, [-1]) for grad in grads], 0)

    return flatgrad


def get_flat(var_list):
    """
    Creates operation that flattens the weights.
    :param var_list: Weights list.
    :return: Operation that flattens the weights.
    """
    return tf.concat([tf.reshape(var, [-1]) for var in var_list], 0)


def discount(x, gamma):
    """
    :param x: Rewards to go list.
    :param gamma: Discount factor.
    :return: Discounted rewards to go.
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def unnormalize_action(action):
    bring = action[0] * myconfig['dlon_std'] + myconfig['dlon_avg']
    speed = action[1] * myconfig['dlat_std'] + myconfig['dlat_avg']
    alt_speed = action[2] * myconfig['dalt_std'] + myconfig['dalt_avg']

    return [bring,speed,alt_speed]

def unnormalize_observation(observation):
    lon = observation[0] * myconfig['longitude_std'] + myconfig['longitude_avg']
    lat = observation[1] * myconfig['latitude_std'] + myconfig['latitude_avg']
    alt = observation[2] * myconfig['altitude_std'] + myconfig['altitude_avg']
    timestamp = observation[3] * myconfig['timestamp_std'] + myconfig['timestamp_avg']
    press_surface = observation[4] * myconfig['Pressure_surface_std'] + myconfig['Pressure_surface_avg']
    rel_humidity_iso = observation[5] * myconfig['Relative_humidity_isobaric_std'] + myconfig['Relative_humidity_isobaric_avg']
    temp_iso = observation[6] * myconfig['Temperature_isobaric_std'] + myconfig['Temperature_isobaric_avg']
    wind_speed_gust_surface = observation[7] * myconfig['Wind_speed_gust_surface_std'] + myconfig['Wind_speed_gust_surface_avg']
    u_component_of_wind_isobaric = observation[8] * myconfig['u-component_of_wind_isobaric_std'] + myconfig['u-component_of_wind_isobaric_avg']
    v_component_of_wind_isobaric = observation[9] * myconfig['v-component_of_wind_isobaric_std'] + myconfig['v-component_of_wind_isobaric_avg']

    return [lon,lat,alt,timestamp,press_surface,rel_humidity_iso,temp_iso,wind_speed_gust_surface,u_component_of_wind_isobaric,v_component_of_wind_isobaric]

def unnormalize_observation2(observation):
    lon = observation[0] * myconfig['longitude_std'] + myconfig['longitude_avg']
    lat = observation[1] * myconfig['latitude_std'] + myconfig['latitude_avg']
    alt = observation[2] * myconfig['altitude_std'] + myconfig['altitude_avg']
    timestamp = observation[3] * myconfig['timestamp_std'] + myconfig['timestamp_avg']

    return [lon,lat,alt,timestamp]

def unnormalize_observation_metar(observation):
    lon = observation[0] * myconfig['longitude_std'] + myconfig['longitude_avg']
    lat = observation[1] * myconfig['latitude_std'] + myconfig['latitude_avg']
    alt = observation[2] * myconfig['altitude_std'] + myconfig['altitude_avg']
    timestamp = observation[3] * myconfig['timestamp_std'] + myconfig['timestamp_avg']
    press_surface = observation[4] * myconfig['Pressure_surface_std'] + myconfig['Pressure_surface_avg']
    rel_humidity_iso = observation[5] * myconfig['Relative_humidity_isobaric_std'] + myconfig['Relative_humidity_isobaric_avg']
    temp_iso = observation[6] * myconfig['Temperature_isobaric_std'] + myconfig['Temperature_isobaric_avg']
    wind_speed_gust_surface = observation[7] * myconfig['Wind_speed_gust_surface_std'] + myconfig['Wind_speed_gust_surface_avg']
    u_component_of_wind_isobaric = observation[8] * myconfig['u-component_of_wind_isobaric_std'] + myconfig['u-component_of_wind_isobaric_avg']
    v_component_of_wind_isobaric = observation[9] * myconfig['v-component_of_wind_isobaric_std'] + myconfig['v-component_of_wind_isobaric_avg']

    drct = observation[10] * myconfig['drct_std'] + myconfig['drct_avg']
    sknt = observation[11] * myconfig['sknt_std'] + myconfig['sknt_avg']
    alti = observation[12] * myconfig['alti_std'] + myconfig['alti_avg']
    vsby = observation[13] * myconfig['vsby_std'] + myconfig['vsby_avg']
    gust = observation[14] * myconfig['gust_std'] + myconfig['gust_avg']

    return [lon,lat,alt,timestamp,press_surface,rel_humidity_iso,temp_iso,wind_speed_gust_surface,u_component_of_wind_isobaric,v_component_of_wind_isobaric, drct, sknt, alti, vsby, gust]


def line_search(loss_f, theta_prev, fullstep, prev_out_run,
                expected_improve_rate, surrogate_run, max_kl):
    """
    TRPO line search.
    :param loss_f: Function that returns surrogate and KL divergence.
    :param theta_prev: Neural networks weights.
    :param fullstep: (H**-1)*g
    :param prev_out_run: Probabilities on actions given by policy with
    theta_prev.
    :param expected_improve_rate: Dot product g.dot((H**-1)*g)
    :param surrogate_run: Surrogate given by policy w.r.t. theta_prev.
    :param max_kl: Maximum KL divergence.
    :return: New weights.
    """
    max_backtracks = 10
    accept_ratio = .1
    for (_n_backtracks, stepfrac) in enumerate(
            .5 ** np.arange(max_backtracks)):
        kl_barrier = max_kl
        theta_new = theta_prev + stepfrac * fullstep
        new_loss = loss_f(theta_new, prev_out_run)
        improve = new_loss[0] - surrogate_run
        expected_improve = expected_improve_rate * stepfrac
        ratio = improve / expected_improve
        mean_kl = np.mean(new_loss[1])

        # if improve >= 0 and mean_kl <= kl_barrier:
        if ratio > accept_ratio:
            # print("Constraints ok. Updating. " ,file=sys.stderr)
            return theta_new, new_loss[0]
    return theta_prev, surrogate_run


class ReplayBuffer(object):

    def __init__(self):
        self.size = 0
        self.observations = []
        self.actions = []

    def seed_buffer(self, observations, actions):
        self.observations = observations
        self.actions = actions
        assert len(observations) == len(actions)
        self.size = len(observations)

    def get_batch(self, batch_size):
        idxs = random.choices(range(self.size), k=batch_size)
        return [self.observations[idx] for idx in idxs], [self.actions[idx] for idx in idxs]
