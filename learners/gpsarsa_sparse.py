
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
#import GPy
from pybrain.rl.agents.logging import LoggingAgent
import numpy as np

from pybrain.datasets.reinforcement import ReinforcementDataSet
from scipy import linalg
from itertools import tee


class GP_SARSA_SPARSE(ValueBasedLearner):
    """ GP State-Action-Reward-State-Action (SARSA) algorithm.
    """


    def __init__(self,gamma=0.99,threshold=-0.5):
        ValueBasedLearner.__init__(self)
        self.thresh=threshold

        self.gamma = gamma

        self.laststate = None
        self.lastaction = None

        self.num_features=2
        self.num_actions=1
        self.kern_c = 10
        self.state_dict = None
        self.cum_reward = np.array([])
        self.u_tilde=np.array([])
        self.C_tilde=np.array([])
        self.d=0.0
        self.v_inv=0.0
        self.c_tild=np.array([])
        self.kern_sigma=0.2
        self.dataset=None
        self.sigma = 1
        self.g=np.array([[]])
        self.K_inv=np.array([[]])
        self.k_tild=np.array([])
        self.k_tild_past=np.array([])
        self.delta=None
        self.delta_k=[]
        self.g_tilde=np.array([])


    def learn(self):

        for seq in self.dataset:
            #seq,foo=tee(seq,2)
            #foo=list(foo)
            #length=len(foo)
            self.laststate = None
            self.lastaction = None
            self.lastreward = None
            c_copy=[]


            for state, action, reward in seq:
                self.k_tild=[]      #not sure, should test with del self.k_tild too
                self.k_tild_past=[]
                temp=None
                c_temp=[]

                if self.laststate is None:

                    if self.state_dict is None:
                        self.state_dict = np.reshape(np.append(state, action),(1, 3))
                        self.K_inv = np.reshape([(1 / self.kernel(np.append(state, action), np.append(state, action)))],(1,1))
                        self.c_tild = np.zeros(1)
                        self.k_tild=np.array([self.kernel(np.append(state,action),np.append(state, action))])
                        self.k_tild = self.k_tild.T
                        self.d=0
                        self.v_inv=0

                        self.g = np.dot(self.K_inv, self.k_tild)
                        self.delta = self.kernel(np.append(state,action),np.append(state, action)) - np.dot(self.k_tild.T, self.g)
                        self.delta=float(self.delta)
                        #print(self.delta)
                        self.g=np.reshape(self.g,(1,1))
                        self.u_tilde=np.zeros(1)
                        self.C_tilde=np.zeros((1,1))
                        '''if (self.delta > self.thresh):
                            self.K_inv = np.hstack(((self.delta * self.K_inv + np.dot(self.g, self.g.T)), -self.g))
                            self.K_inv = np.vstack((self.K_inv,np.append(self.g.T,[1]))
                            self.K_inv = (self.K_inv / self.delta)
                            self.g = np.zeros((1,1))
                            self.u_tilde = np.zeros(1)
                            self.C_tilde = np.zeros((1,1))
                            self.c_tild=np.zeros(1)
                        '''
                        self.lastaction = action
                        self.laststate = state
                        self.lastreward = reward
                        continue

                    else:
                        self.c_tild = np.zeros(self.state_dict.shape[0])
                        self.d=0
                        self.v_inv=0
                        for elem in range(self.state_dict.shape[0]):
                            self.k_tild = np.append(self.k_tild,[self.kernel(self.state_dict[elem], np.append(state, action))])
                            self.k_tild = self.k_tild.T
                        self.g = np.dot(self.K_inv, self.k_tild)
                        self.delta = self.kernel(np.append(state, action), np.append(state, action)) - np.dot(self.k_tild.T,self.g)
                        if (self.delta > self.thresh):
                            self.state_dict=np.vstack((self.state_dict,np.append(state,action)))
                            self.K_inv = np.hstack(((self.delta * self.K_inv + np.dot(self.g, self.g.T)), -self.g))
                            self.K_inv = np.vstack((self.K_inv, np.append(self.g.T, [1])))
                            self.K_inv = self.K_inv * (1 / self.delta)
                            self.g =np.append(np.zeros(self.state_dict.shape[0]-1),[1]).T
                            self.u_tilde = np.vstack((self.u_tilde, [0]))
                            self.C_tilde = np.hstack((self.C_tilde, np.zeros((self.C_tilde.shape[1], 1))))
                            self.C_tilde = np.vstack((self.C_tilde,np.zeros(self.C_tilde.shape[1]))) #correction
                            self.c_tild = np.vstack((self.c_tild, [0]))
                        self.lastaction = action
                        self.laststate = state
                        self.lastreward = reward
                        continue

                else:
                    for num in range(self.state_dict.shape[0]):
                        self.k_tild = np.append(self.k_tild,[self.kernel(self.state_dict[num], np.append(state, action))])
                        self.k_tild=self.k_tild.T
                        self.k_tild_past=np.append(self.k_tild_past,[self.kernel(self.state_dict[num], np.append(self.laststate, self.lastaction))])
                        self.k_tild_past = self.k_tild_past.T
                    #print(self.k_tild)
                    #print(self.k_tild_past)
                    self.g_tilde=np.dot(self.K_inv,self.k_tild)
                    self.g_tilde=np.reshape(self.g_tilde,(self.K_inv.shape[0],1))
                    self.delta=self.kernel(np.append(state, action), np.append(state, action))-np.dot(self.k_tild.T,self.g_tilde)
                    self.delta_k=(self.k_tild_past-self.gamma*self.k_tild)

                    #print(self.g_tilde)
                    #self.delta_k=self.delta_k.T
                    self.d=np.array([self.gamma*self.sigma*self.v_inv*self.d]) + reward -np.dot(self.delta_k.T,self.u_tilde)
                    self.d=float(self.d)

                    if(self.delta>self.thresh):
                        self.state_dict = np.vstack((self.state_dict, np.append(state, action)))

                        self.K_inv=np.hstack(((self.delta*self.K_inv+np.dot(self.g_tilde,self.g_tilde.T)),-self.g_tilde))
                        self.K_inv = np.vstack((self.K_inv, np.append(self.g_tilde.T, [1])))
                        self.K_inv=self.K_inv*(1/self.delta)

                        if(len(self.k_tild)-1>0):
                            self.g_tilde=np.append(np.zeros(len(self.k_tild)-1),[1])
                            self.g_tilde=self.g_tilde.T
                        else:
                            self.g_tilde=np.array([0])


                        self.h_tilde=np.append(self.g_tilde.T,[-self.gamma])
                        self.h_tilde=self.h_tilde.T                                 #todo:doubt, g_tilde or g
                        #print(self.k_tild_past-2*self.gamma*self.k_tild)
                        #print(self.g_tilde)
                        self.ktt=np.dot(self.g_tilde.T,(self.k_tild_past-2*self.gamma*self.k_tild))   #todo : same above
                        self.ktt=float(self.ktt)+(self.gamma**2)*(self.kernel(np.append(state,action),np.append(state,action)))

                        temp=self.v_inv
                        self.v_inv=-(self.gamma**2)*(self.sigma**2)*self.v_inv + (2 * self.gamma * self.v_inv * self.sigma)*np.dot(self.c_tild,self.delta_k.T)
                        self.v_inv+=self.ktt-np.dot(self.delta_k.T,np.dot(self.C_tilde,self.delta_k)) + (1+self.gamma**2)*self.sigma
                        self.v_inv=float(self.v_inv)
                        self.v_inv=(1/self.v_inv)

                        self.c_tild=(self.gamma*self.sigma*temp)*np.append(self.c_tild,[0])

                        self.c_tild+=self.h_tilde.T-np.append(np.dot(self.C_tilde,self.delta_k),[0])
                        print(self.c_tild)

                        self.u_tilde=np.vstack((self.u_tilde,[0]))
                        self.C_tilde=np.hstack((self.C_tilde,np.zeros((self.C_tilde.shape[1],1))))
                        self.C_tilde = np.vstack((self.C_tilde, np.zeros(self.C_tilde.shape[1])))

                    else:
                        c_temp=self.c_tild
                        self.h_tilde=self.g-self.gamma*self.g_tilde
                        self.c_tild=(self.gamma*self.sigma*self.v_inv)*self.c_tild + self.h_tilde - np.dot(self.C_tilde,self.delta_k)
                        self.v_inv=(1+self.gamma**2)*self.sigma+np.dot(self.delta_k.T,(self.c_tild+(self.gamma*self.sigma*self.v_inv)*c_temp)) - (self.gamma**2)*(self.sigma**2)*self.v_inv
                        self.v_inv=float(self.v_inv)
                        self.v_inv=(1/self.v_inv)

                    self.laststate=state
                    self.lastaction=action

            self.g_tilde=np.zeros(self.state_dict.shape[0])
            self.delta=0
            for some in range(self.state_dict.shape[0]):
                self.delta_k = np.append([],[self.kernel(self.state_dict[some], np.append(self.laststate, self.lastaction))])
            self.delta_k = self.delta_k.T
            c_copy=self.c_tild
            self.v_inv=self.sigma**2+np.dot(self.delta_k.T,(self.c_tild+(self.gamma*self.sigma*self.v_inv)*c_copy)) - (self.gamma**2)*(self.sigma**2)*self.v_inv
            self.v_inv=float(self.v_inv)
            self.v_inv=(1/self.v_inv)

            self.u_tilde=self.u_tilde+self.c_tild*self.d*self.v_inv
            self.C_tilde=self.C_tilde+self.v_inv*self.c_tild*self.c_tild.T
            self.g=self.g_tilde
            self.update_posterior(self.u_tilde,self.C_tilde)
            print('mean',self.u_tilde)

        #print(self.H)
        print(self.state_dict.shape)



    def action_kern(self,act1,act2):  #delta kernel
        if(act1==act2):
            return 1
        else:
            return 0


    def state_kern(self,state1,state2):
        return(self.kern_c*np.exp(-(np.sqrt(np.sum(np.subtract(state1,state2)**2))/(2*self.kern_sigma**2))))  #todo: if we use GPy kernel, product can't be multiplied

    def kernel(self,stat1,stat2):
        return(self.state_kern(stat1[0:2],stat2[0:2])*self.action_kern(stat1[2],stat2[2]))

    def ret_dict(self):
        return self.state_dict

    '''def ret_reward(self):
        return self.cum_reward
    '''
    def update_posterior(self,mean, covariance):
        return mean,covariance

















