
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
import GPy
from pybrain.rl.agents.logging import LoggingAgent
import numpy as np
from numpy.linalg import inv
from pybrain.datasets.reinforcement import ReinforcementDataSet
from scipy import linalg
#from copy import copy
from itertools import tee
class GP_SARSA(ValueBasedLearner):
    """ GP State-Action-Reward-State-Action (SARSA) algorithm.
    """


    def __init__(self,gamma=0.99):
        ValueBasedLearner.__init__(self)


        self.gamma = gamma

        self.laststate = None
        self.lastaction = None

        self.num_features=2
        self.num_actions=1
        self.kern_c = 10

        self.covariance_mat=np.array([[]])
        self.inv=np.array([])
        self.state_dict = None
        self.cum_reward = np.array([])
        self.H=[]
        self.kern_sigma=0.2
        self.dataset=None
        self.sigma = 1



    def learn(self):
         #initialise K for first episode



        for seq in self.dataset:
            #seq,foo=tee(seq,2)
            #foo=list(foo)
            #print("the length is",foo[0])
            # information from the previous episode (sequence)
            # should not influence the training on this episode
            self.laststate = None
            self.lastaction = None
            self.lastreward = None

            for state, action, reward in seq:

                if self.state_dict is None:

                    self.lastaction = action
                    self.laststate = state
                    self.lastreward = reward
                    #self.covariance_list=np.array([self.kernel(np.append(self.laststate,self.lastaction),np.append(state,action))])
                    self.state_dict = np.reshape(np.append(self.laststate, self.lastaction), (1, 3))
                    self.cum_reward = np.append(self.cum_reward, reward)


                    for num in range(self.state_dict.shape[0]):
                        self.covariance_list = np.array([[self.kernel(self.state_dict[num],np.append(state,action))]])
                    self.covariance_mat = self.covariance_list
                    #self.covariance_mat = np.hstack((self.covariance_mat, self.covariance_list))
                    #self.covariance_mat = np.vstack((self.covariance_mat, np.append(self.covariance_list, self.kern_c * 1)))
                    print('ist step')
                    continue

                else:

                    self.cum_reward = np.append(self.cum_reward, reward)
                    self.covariance_list=[]
                    for element in range(self.state_dict.shape[0]):
                        self.covariance_list = np.append(self.covariance_list, [self.kernel(self.state_dict[element],np.append(state, action))])
                    self.covariance_list = np.reshape(self.covariance_list, (1, self.covariance_list.shape[0]))
                    self.state_dict = np.append(self.state_dict, np.reshape(np.append(state, action), (1, 3)), axis=0)
                    #self.H = np.append(self.H, np.zeros((self.H.shape[0], 1)), axis=1)
                    #self.H = np.append(self.H, np.append(np.zeros((1, self.H.shape[0])), [[1., -self.gamma]], axis=1),axis=0)
                    self.covariance_mat = np.append(self.covariance_mat, self.covariance_list.transpose(), axis=1)
                    self.covariance_mat = np.vstack((self.covariance_mat, np.append(self.covariance_list, [self.kernel(np.append(state, action),np.append(state, action))])))
                    element=0
                    #print(self.state_dict, self.state_dict.shape
                    self.laststate=state
                    self.lastaction=action
                    self.lastreward=reward

            self.update_inv(self.covariance_mat,self.get_H(self.state_dict.shape[0]))

        #print(self.H)
        print(self.state_dict.shape)
        #print(self.covariance_mat.shape)


    def action_kern(self,act1,act2):  #delta kernel
        if(act1==act2):
            return 1
        else:
            return 0


    def state_kern(self,state1,state2):
        return(self.kern_c*np.exp(-(np.sqrt(np.sum(np.subtract(state1,state2)**2))/(2*self.kern_sigma**2))))  #todo: if we use GPy kernel, product can't be multiplied

    def kernel(self,stat1,stat2):
        return(self.state_kern(stat1[0:2],stat2[0:2])*self.action_kern(stat1[2],stat2[2]))


    def q_init(self,inistate,iniaction):
        return (np.sqrt(self.kernel(inistate,iniaction))*np.random.randn())


    def update_inv(self,K,H):
        self.sigma=0.05 #noise variance
        #self.inv = np.dot(np.dot(k, H.transpose()),np.dot(inv(np.dot(np.dot(H, K), H.transpose()) + sigma ** 2 * np.dot(H, H.transpose())), R))
        self.inv=np.dot(H.transpose(),linalg.inv(np.dot(np.dot(H,K),H.transpose())+self.sigma**2*np.dot(H,H.transpose())))
        #print(self.inv.shape)

    def ret_h(self):
        return self.H

    def ret_dict(self):
        return self.state_dict

    def ret_reward(self):
        return self.cum_reward

    def ret_cov(self):
        return self.covariance_mat

    def get_H(self,dim):
        self.H=np.eye(dim,dim)
        for elem in range(dim-1):
            self.H[elem,elem+1]=-self.gamma
        return self.H















