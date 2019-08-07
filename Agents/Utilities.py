import scipy.signal
import numpy as np

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
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    return np.sum(rewards) * (1./len(lengths))
