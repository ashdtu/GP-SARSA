from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
import numpy as np

class GP_SARSA_SPARSE(ValueBasedLearner):
    """ GP State-Action-Reward-State-Action (SARSA) algorithm.

    Offline Episodic Learner"""


    def __init__(self,gamma=0.95,threshold=0):
        ValueBasedLearner.__init__(self)
        self.num_features = 5
        self.num_actions = 1

        self.sigma = 1.0
        self.kern_c = 10
        self.kern_sigma = 0.5

        self.thresh=threshold
        self.gamma = gamma

        self.laststate = None
        self.lastaction = None
        self.lastreward=None


        self.state_dict = None
        self.cum_reward = np.array([])
        self.u_tilde=np.array([])
        self.C_tilde=np.array([[]])
        self.d=0.0
        self.v_inv=0.0
        self.c_tild=np.array([])
        self.dataset=None
        #self.g=np.array([])
        self.K_inv=np.array([[]])
        #self.k_tild=np.array([])
        #self.k_tild_past=np.array([])
        #self.delta=None
        #self.delta_k=[]
        #self.g_tilde=np.array([])

    def learn(self):

        for seq in self.dataset:
            self.laststate = None
            self.lastaction = None
            self.lastreward = None
            d = 0
            v_inv = 0

            for state, action, reward in seq:

                if self.laststate is None:

                    if self.state_dict is None:
                        self.state_dict = np.reshape(np.append(state, action), (1, self.num_features + 1))
                        self.K_inv = np.array([[(1 / self.kernel(np.append(state, action), np.append(state, action)))]])
                        self.u_tilde = np.zeros(1)
                        self.C_tilde = np.zeros((1, 1))

                        g = np.ones(1)
                        c = np.zeros(1)
                        d = 0
                        v_inv = 0

                        self.laststate = state
                        self.lastaction = action
                        self.lastreward = reward
                        continue

                    elif self.state_dict is not None and self.laststate is None:
                        c = np.zeros(self.state_dict.shape[0])
                        k_tilde = self.cov_list(state, action)
                        g = np.dot(self.K_inv, k_tilde[:, np.newaxis])
                        some_temp = g
                        g = np.reshape(g, g.shape[0])
                        delta = self.kernel(np.append(state, action), np.append(state, action)) - np.dot(k_tilde, g)
                        delta = float(delta)

                        if (delta > self.thresh):
                            self.state_dict = np.vstack((self.state_dict, np.append(state, action)))
                            self.K_inv = np.hstack(((delta * self.K_inv + np.dot(some_temp, some_temp.T)), -some_temp))
                            self.K_inv = np.vstack((self.K_inv, np.append(-some_temp.T, [1])))
                            self.K_inv = self.K_inv * (1 / delta)
                            g = np.append(np.zeros(self.state_dict.shape[0] - 1), [1])
                            self.u_tilde = np.append(self.u_tilde, [0])
                            self.C_tilde = np.hstack((self.C_tilde, np.zeros((self.C_tilde.shape[0], 1))))
                            self.C_tilde = np.vstack((self.C_tilde, np.zeros(self.C_tilde.shape[1])))  # correction
                            c = np.append(c, [0])

                        self.lastaction = action
                        self.laststate = state
                        self.lastreward = reward
                        continue

                k_tilde = self.cov_list(state, action)
                k_tilde_past = self.cov_list(self.laststate, self.lastaction)

                g_tilde = np.dot(self.K_inv, k_tilde)
                temp = g_tilde[:, np.newaxis]

                delta = self.kernel(np.append(state, action), np.append(state, action)) - np.dot(k_tilde, g_tilde)
                delta = float(delta)
                # self.delta_list.append(self.delta)

                delta_k = k_tilde_past - self.gamma * k_tilde

                d = (self.gamma * self.sigma * v_inv * d) + self.lastreward - np.dot(delta_k,self.u_tilde)
                d = float(d)

                if (delta > self.thresh):
                    self.state_dict = np.vstack((self.state_dict, np.append(state, action)))

                    self.K_inv = np.hstack(((delta * self.K_inv + np.dot(temp, temp.T)), -temp))
                    self.K_inv = np.vstack((self.K_inv, np.append(-temp.T, [1])))
                    self.K_inv = self.K_inv * (1 / delta)

                    g_tilde = np.append(np.zeros(self.state_dict.shape[0] - 1), [1])
                    h_tilde = np.append(g, [-self.gamma])

                    ktt = np.dot(g, (k_tilde_past - 2 * self.gamma * k_tilde))
                    ktt = float(ktt) + (self.gamma ** 2) * (
                    self.kernel(np.append(state, action), np.append(state, action)))

                    c_tild = (self.gamma * self.sigma * v_inv) * np.append(c, [0])
                    c_tild += h_tilde - np.append(np.dot(self.C_tilde, delta_k), [0])

                    v_inv = -(self.gamma ** 2) * (self.sigma ** 2) * v_inv + (
                            2 * self.gamma * v_inv * self.sigma) * np.dot(c, delta_k)

                    v_inv += ktt - np.dot(delta_k, np.dot(self.C_tilde, delta_k)) + (1 + self.gamma ** 2) * self.sigma

                    v_inv = float(v_inv)
                    v_inv = (1 / v_inv)

                    self.u_tilde = np.append(self.u_tilde, [0])

                    self.C_tilde = np.hstack((self.C_tilde, np.zeros((self.C_tilde.shape[0], 1))))
                    self.C_tilde = np.vstack((self.C_tilde, np.zeros(self.C_tilde.shape[1])))

                else:
                    h_tilde = g - self.gamma * g_tilde
                    c_tild = (self.gamma * self.sigma * v_inv) * c + h_tilde - np.reshape(
                        np.dot(self.C_tilde, delta_k[:, np.newaxis]), self.C_tilde.shape[0])

                    v_inv = (1+self.gamma**2)*self.sigma + np.dot(delta_k, (c_tild + (self.gamma * self.sigma * v_inv) * c)) - (
                            self.gamma ** 2) * (self.sigma ** 2) * v_inv
                    v_inv = float(v_inv)
                    v_inv = (1 / v_inv)

                self.u_tilde = self.u_tilde + c_tild * d * v_inv
                c_temp = c_tild[:, np.newaxis]
                self.C_tilde = self.C_tilde + v_inv * np.dot(c_temp, c_temp.T)
                c = c_tild
                g = g_tilde
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward

            g_tilde = np.zeros(self.state_dict.shape[0])
            delta_k = self.cov_list(self.laststate, self.lastaction)
            d = (self.gamma * self.sigma * v_inv * d) + self.lastreward - np.dot(delta_k, self.u_tilde)
            d = float(d)

            h_tilde = g - self.gamma * g_tilde
            c_tild = (self.gamma * self.sigma * v_inv) * c + h_tilde - np.reshape(
                np.dot(self.C_tilde, delta_k[:, np.newaxis]), self.C_tilde.shape[0])

            v_inv = self.sigma + np.dot(delta_k, (c_tild + (self.gamma * self.sigma * v_inv) * c)) - (
                    self.gamma ** 2) * (self.sigma ** 2) * v_inv
            v_inv = float(v_inv)
            v_inv = (1 / v_inv)

            self.u_tilde = self.u_tilde + c_tild * d * v_inv
            min_temp = c_tild[:, np.newaxis]
            self.C_tilde = self.C_tilde + v_inv * np.dot(min_temp, min_temp.T)

    def action_kern(self, act1, act2):  # delta kernel
        if (act1 == act2):
            return 1
        else:
            return 0

    def state_kern(self, state1, state2):
        a = self.kern_c * np.exp(-(np.sqrt(np.sum(np.subtract(state1, state2) ** 2)) / (2 * self.kern_sigma ** 2)))
        # print('kernel',a)
        return (self.kern_c * np.exp(-(np.sqrt(np.sum(np.subtract(state1, state2) ** 2)) / (
                2 * self.kern_sigma ** 2))))  # todo: if we use GPy kernel, product can't be multiplied

    def kernel(self, stat1, stat2):
        return (self.state_kern(stat1[0:self.num_features], stat2[0:self.num_features]) * self.action_kern(
            stat1[self.num_features], stat2[self.num_features]))

    def ret_dict(self):
        return self.state_dict

    def update_posterior(self, mean, covariance):
        return mean, covariance

    def cov_list(self, inp_state, inp_act):
        temp_list = np.array([])
        for _ in range(self.state_dict.shape[0]):
            temp_list = np.append(temp_list, [self.kernel(self.state_dict[_], np.append(inp_state, inp_act))])
        return temp_list.T
