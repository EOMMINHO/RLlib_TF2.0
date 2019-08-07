import tensorflow as tf
import numpy as np
import gym
import Utilities
import time
import pdb


class VPG(object):
    def __init__(self, env_name, train_step=5000, render_step=1000, gamma=0.99, lamb=0.97, v_iters=80, epochs=50):
        self.gamma = gamma
        self.lamb = lamb
        self.v_iter = v_iters
        self.epochs = epochs

        self.env = gym.make(env_name)
        self.train_step = train_step
        self.render_step = render_step
        self.value_function = None
        self.policy_function = None

        self.policy_optimizer = tf.keras.optimizers.Adam()
        self.value_optimizer = tf.keras.optimizers.Adam()

    def set_value_function(self, function):
        self.value_function = function

    def set_policy_function(self, function):
        self.policy_function = function

    def train_value(self, statebuffer, returnbuffer):
        # fit value function to the return buffer
        for k in range(self.v_iter):
            with tf.GradientTape() as tape:
                # Update value via gradient descent
                value_buf = self.value_function(statebuffer)
                mse = np.square(value_buf.reshape([-1]) - returnbuffer) * (1 / len(returnbuffer))
                loss2 = tf.reduce_sum(mse)
            value_gradient = tape.gradient(loss2, self.value_function.trainable_variables)
            self.value_optimizer.apply_gradients(zip(value_gradient, self.value_function.trainable_variables))

    def train_step(self):
        # For k steps do the following
        # collect set of trajectories
        with tf.GradientTape(persistent=True) as tape:
            statebuffer = []
            actionbuffer = []
            rewardbuffer = []
            lengthbuffer = []  # stores the length of each episode
            llhbuffer = []  # stores the log likelihood of the action
            length = 0
            observation = self.env.reset()
            statebuffer.append(observation)
            for i in range(self.train_step):
                # with batch 1 size observation
                observation = tf.reshape(observation, [1, observation.shape[0]])
                # with policy function sample an action
                action_prob = self.policy_function(observation)
                action_prob = tf.squeeze(action_prob)
                action = np.random.choice(action_prob.shape[0], p=action_prob)  # sampling from distribution
                likelihood = action_prob[action]
                llhbuffer.append(likelihood)
                # for the given s, a, output s', r
                observation, reward, done, _ = self.env.step(action)
                # record the state and action
                statebuffer.append(observation)  # shape: [total step, state dimension]
                actionbuffer.append(action)  # shape: [total step, ]
                rewardbuffer.append(float(reward))  # shape: [total step, ]
                length += 1

                if done or (i == self.train_step - 1):
                    if done:
                        lengthbuffer.append(length)
                        statebuffer.pop()  # delete the terminated state
                        observation = self.env.reset()
                        statebuffer.append(observation)
                    else:
                        total_length = sum(lengthbuffer)
                        statebuffer = statebuffer[:total_length]
                        llhbuffer = llhbuffer[:total_length]
                        actionbuffer = actionbuffer[:total_length]
                        rewardbuffer = rewardbuffer[:total_length]
                        pass
                    length = 0

            pdb.set_trace()
            # compute rewards to go
            returnbuffer = Utilities.discount_cumsum_trun(rewardbuffer, self.gamma, lengthbuffer)
            # compute advantage estimates
            value_buf = self.value_function(statebuffer)
            value_buf_ep = np.append(value_buf, [[0]],
                                     axis=0)  # only for episodic environment, make the terminated value 0
            delta_buf = np.asarray(returnbuffer) + self.gamma * value_buf_ep[1:] - value_buf_ep[:-1]
            advantage_buf = Utilities.discount_cumsum_trun(delta_buf, self.gamma * self.lamb, lengthbuffer)
            # compute the log likelihood
            llhbuffer = np.log(llhbuffer)

        # Update policy via gradient ascent
        loss1 = -tf.reduce_sum(llhbuffer * advantage_buf) * (1. / len(lengthbuffer))
        policy_gradient = tape.gradient(loss1, self.policy_function.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradient, self.policy_function.trainable_variables))
        del tape

        # fit the value function to mean-squared error
        self.train_value(statebuffer, returnbuffer)

        # the average return (undiscounted)
        avg_return = Utilities.reward2avg(rewardbuffer, lengthbuffer)

        return avg_return

    def train(self):
        for epoch in self.epochs:
            start = time.time()
            avg_return = self.train_step()
            time_elap = time.time() - start
            print("Time for epoch {}, average return {}".format(time_elap, avg_return))

    def render(self):
        """
        Render the agents moves to see how it works
        :return: None
        """
        observation = self.env.reset()
        for i in range(self.render_step):
            self.env.render()

            observation = tf.reshape(observation, [1, observation.shape[0]])
            # print("observation is {}".format(observation))
            action_prob = self.policy_function(observation)
            # print("action_prob is {}".format(action_prob))
            action_prob = tf.squeeze(action_prob)
            # print("action_prob is {}".format(action_prob))
            action = np.random.choice(action_prob.shape[0], p=action_prob)
            observation, reward, done, info = self.env.step(action)

            if done:
                observation = self.env.reset()
        self.env.close()


class value_function(tf.keras.Model):
    """
    Return the value of that state for the given observation
    """

    def __init__(self):
        super(value_function, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(10)
        self.dense3 = tf.keras.layers.Dense(1)

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x


class policy_function(tf.keras.Model):
    """
    Return the probability of action for the given observation
    """

    def __init__(self, output_dim):
        super(policy_function, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(10)
        self.dense3 = tf.keras.layers.Dense(output_dim)

        self.relu = tf.keras.layers.ReLU()
        self.softmax = tf.keras.layers.Softmax()
        pass

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    vf = value_function()
    pf = policy_function(2)
    agent = VPG("CartPole-v1")
    agent.set_value_function(vf)
    agent.set_policy_function(pf)
    #agent.render()
