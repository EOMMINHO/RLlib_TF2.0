import scipy.signal
import numpy as np
import tensorflow as tf
import pdb

@tf.function
def obs2action(policy_function, obs):
    batch = tf.expand_dims(obs, 0)
    action_logit = policy_function(batch)
    action_logit = tf.squeeze(action_logit)
    action_prob = tf.nn.softmax(action_logit)
    action = tf.random.categorical(tf.reshape(action_logit, [1, -1]), 1)[0][0]  # sampling from distribution

    likelihood = action_prob[action]
    llh = tf.math.log(likelihood + 1e-8)

    # return as a one-hot vector
    action_oh = tf.one_hot(action, depth=2)

    return action_oh, action, llh

@tf.function
def obs2value(value_function, obs):
    batch = tf.expand_dims(obs, 0)
    value = value_function(batch)
    value = tf.squeeze(value)
    return value

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_cumsum_trun(x, discount, length):
    """
    compute discounted cumulative sums of vectors.
    truncate x in length array
    :param x:
        vector x,
        [x0,
         x1,
         x2,
         x3,
         x4]
    :param length:
        vector length,
        [3,
         2]
    :return:
        truncated by the vector length
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    ret_arr = x.copy()
    total_len = 0
    for len in length:
        tmp_list = ret_arr[total_len : total_len + len]
        ret_arr[total_len: total_len + len] = discount_cumsum(tmp_list, discount)
        total_len += len
    return ret_arr

def reward2avg(rewards, lengths):
    """
    for episodic environment, change overall rewards to the average reward for each episode
    :param rewards: the vector of rewards
    :param length: the vector of lengths for each episode
    :return: the average undiscounted return
    """
    #rewards = np.array(rewards)
    #lengths = np.array(lengths)
    total_length = -1
    for length in lengths:
        total_length += length + 1
        rewards[total_length] = 0

    return sum(rewards) * (1./len(lengths))

def makedelta(rew_buf, val_buf, len_buf, gamma):
    """
    make a delta from three buffers
    :param rew_buf: numpy array of rewards
    :param val_buf: numpy array of values
    :param len_buf: length of states
    :return: numpy array of deltas
    """
    total_length = 0
    delta_buf = []
    for length in len_buf:
        tmp_val = val_buf[total_length:total_length+length]
        tmp_ret = rew_buf[total_length:total_length+length]
        tmp_delta = tmp_ret[:-1] + gamma * tmp_val[1:] - tmp_val[:-1]
        delta_buf.extend(tmp_delta)
    return np.array(delta_buf)

def rewval2advantage(rew_buf, val_buf, len_buf, gamma, lamb):
    """
    compute approximation of advantage given rewards, values, lengths, and discounts
    :param rew_buf: numpy array of rewards
    :param val_buf: numpy array of values
    :param len_buf: numpy array of lengths
    :param gamma: discount hyperparameter
    :param lamb: discount hyperparameter
    :return: numpy array of approximated advantages
    """
    delta_buf = makedelta(rew_buf, val_buf, len_buf, gamma)
    # decrement length buffer
    len_buf = np.array(len_buf) - 1
    return discount_cumsum_trun(delta_buf, gamma * lamb, len_buf)

def rewstate2advantage(rew_buf, state_buf, len_buf, gamma, lamb, value_function):
    """
    compute approximation of advantage given rewards, states, lengths, and discounts
    :param rew_buf:
    :param state_buf:
    :param len_buf:
    :param gamma:
    :param lamb:
    :return:
    """
    val_buf = value_function(state_buf)  # shape: (steps, 1)
    val_buf = tf.reshape(val_buf, [-1])  # squeeze to shape: (steps, )
    return rewval2advantage(rew_buf, val_buf, len_buf, gamma, lamb)


if __name__ == "__main__":
    rew_buf = np.array([1., 1., 1., 1., 1., 1.])
    val_buf = np.array([0.3, 0.2, 0.4, 0.2, 0.3, 0.1])
    gamma = 0.5
    lamb = 0.5
    len_buff = np.array([4, 2])
    print(rewval2advantage(rew_buf, val_buf, len_buff, gamma, lamb))