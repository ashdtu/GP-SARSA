
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
import numpy as np
class GP_SARSA_SPARSE(ValueBasedLearner):
    """ GP State-Action-Reward-State-Action (SARSA) algorithm.
    """


    def __init__(self,gamma=0.99,threshold=10):
        ValueBasedLearner.__init__(self)
        self.thresh=threshold

        self.gamma = gamma

        self.laststate = None
        self.lastaction = None

        self.num_features=9
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
        self.sigma = 0.5
        self.g=np.array([])
        self.K_inv=np.array([[]])
        self.k_tild=np.array([])
        self.k_tild_past=np.array([])
        self.delta=None
        self.delta_k=[]
        self.g_tilde=np.array([])



    def learn(self):
        self.delta_list = []
        for seq in self.dataset:

            self.laststate = None
            self.lastaction = None
            self.lastreward = None



            for state, action, reward in seq:

                if self.laststate is None:

                    if self.state_dict is None:
                        self.state_dict = np.reshape(np.append(state, action),(1, 10))
                        self.K_inv = np.reshape([(1 / self.kernel(np.append(state, action), np.append(state, action)))],(1,1))
                        self.c = np.zeros(1)
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
                        self.lastaction = action
                        self.laststate = state
                        self.lastreward = reward
                        continue

                    else:
                        self.c = np.zeros(self.state_dict.shape[0])
                        self.d=0
                        self.v_inv=0
                        self.k_tild=self.cov_list(state,action)
                        self.g = np.dot(self.K_inv, self.k_tild)
                        some_temp=np.reshape(self.g,(self.state_dict.shape[0],1))
                        self.delta = self.kernel(np.append(state, action), np.append(state, action)) - np.dot(self.k_tild.T,self.g)
                        self.delta=float(self.delta)
                        self.delta_list.append(self.delta)
                        if (self.delta > self.thresh):
                            self.state_dict=np.vstack((self.state_dict,np.append(state,action)))
                            self.K_inv = np.hstack(((self.delta*self.K_inv + np.dot(some_temp,some_temp.T)), -some_temp))
                            self.K_inv = np.vstack((self.K_inv, np.append(-some_temp.T, [1])))
                            self.K_inv = self.K_inv * (1 / self.delta)
                            self.g =np.append(np.zeros(self.state_dict.shape[0]-1),[1]).T
                            self.u_tilde = np.append(self.u_tilde, [0])
                            self.C_tilde = np.hstack((self.C_tilde, np.zeros((self.C_tilde.shape[0], 1))))
                            self.C_tilde = np.vstack((self.C_tilde,np.zeros(self.C_tilde.shape[1]))) #correction
                            self.c = np.append(self.c, [0])
                        self.lastaction = action
                        self.laststate = state
                        self.lastreward = reward
                        continue

                else:

                    self.k_tild=self.cov_list(state,action)
                    self.k_tild_past=self.cov_list(self.laststate,self.lastaction)
                    self.g_tilde=np.dot(self.K_inv,self.k_tild)

                    temp=np.reshape(self.g_tilde,(self.state_dict.shape[0],1))
                    self.delta=self.kernel(np.append(state, action), np.append(state, action))-np.dot(self.k_tild.T,self.g_tilde)
                    self.delta=float(self.delta)
                    self.delta_list.append(self.delta)
                    self.delta_k=self.k_tild_past-self.gamma*self.k_tild

                    self.d=(self.gamma*self.sigma*self.v_inv*self.d) + reward -np.dot(self.delta_k,self.u_tilde)
                    self.d=float(self.d)

                    if(self.delta>self.thresh):
                        self.state_dict = np.vstack((self.state_dict, np.append(state, action)))

                        self.K_inv=np.hstack(((self.delta*self.K_inv+np.dot(temp,temp.T)),-temp))
                        self.K_inv = np.vstack((self.K_inv, np.append(-temp.T, [1])))
                        self.K_inv=self.K_inv*(1/self.delta)

                        self.g_tilde=np.append(np.zeros(self.state_dict.shape[0]-1),[1])
                        self.g_tilde=self.g_tilde.T

                        self.h_tilde=np.append(self.g.T,[-self.gamma])
                        self.h_tilde=self.h_tilde.T


                        self.ktt=np.dot(self.g.T,(self.k_tild_past-2*self.gamma*self.k_tild))
                        self.ktt=float(self.ktt)+(self.gamma**2)*(self.kernel(np.append(state,action),np.append(state,action)))

                        self.c_tild = (self.gamma * self.sigma * self.v_inv) * np.append(self.c, [0])
                        self.c_tild = self.c_tild + self.h_tilde.T - np.append(np.dot(self.C_tilde, self.delta_k), [0])


                        self.v_inv= -(self.gamma**2)*(self.sigma**2)*self.v_inv + (2 * self.gamma * self.v_inv * self.sigma)*np.dot(self.c,self.delta_k)
                        self.v_inv+=self.ktt-np.dot(self.delta_k.T,np.dot(self.C_tilde,self.delta_k)) + (1+self.gamma**2)*self.sigma
                        self.v_inv=float(self.v_inv)
                        self.v_inv=(1/self.v_inv)

                        self.u_tilde=np.append(self.u_tilde,[0])

                        self.C_tilde=np.hstack((self.C_tilde,np.zeros((self.C_tilde.shape[0],1))))
                        self.C_tilde = np.vstack((self.C_tilde, np.zeros(self.C_tilde.shape[1])))

                    else:

                        self.h_tilde=self.g.T-self.gamma*self.g_tilde


                        self.c_tild=(self.gamma*self.sigma*self.v_inv)*self.c + self.h_tilde.T - np.transpose(np.dot(self.C_tilde,self.delta_k))
                        self.v_inv=(1+self.gamma**2)*self.sigma+np.dot(self.delta_k.T,(self.c_tild.T+(self.gamma*self.sigma*self.v_inv)*self.c.T)) - (self.gamma**2)*(self.sigma**2)*self.v_inv

                        self.v_inv = (1 / self.v_inv)
                        self.v_inv=float(self.v_inv)


                self.u_tilde = self.u_tilde + self.c_tild * self.d * self.v_inv
                c_temp=np.reshape(self.c_tild,(len(self.c_tild),1))
                self.C_tilde = self.C_tilde + self.v_inv * np.dot(c_temp,c_temp.T)

                self.c=self.c_tild
                self.g=self.g_tilde
                self.laststate=state
                self.lastaction=action

                #print('dict',self.state_dict.shape)




            self.g_tilde=np.zeros(self.state_dict.shape[0])
            self.delta=0
            self.delta_k=self.cov_list(self.laststate,self.lastaction)
            self.h_tilde = self.g.T - self.gamma * self.g_tilde
            self.c_tild=(self.gamma*self.sigma*self.v_inv)*self.c + self.h_tilde.T - np.transpose(np.dot(self.C_tilde,self.delta_k))
            self.v_inv=self.sigma+np.dot(self.delta_k.T,(self.c_tild+(self.gamma*self.sigma*self.v_inv)*self.c)) - (self.gamma**2)*(self.sigma**2)*self.v_inv
            self.v_inv=float(self.v_inv)
            self.v_inv=(1/self.v_inv)
            self.u_tilde = self.u_tilde + self.c_tild * self.d * self.v_inv
            min_temp=np.reshape(self.c_tild,(len(self.c_tild),1))
            self.C_tilde = self.C_tilde + self.v_inv * np.dot(min_temp,min_temp.T)




    def action_kern(self,act1,act2):  #delta kernel
        if(act1==act2):
            return 1
        else:
            return 0

    def state_kern(self,state1,state2):
        return(self.kern_c*np.exp(-(np.sqrt(np.sum(np.subtract(state1,state2)**2))/(2*self.kern_sigma**2))))  #todo: if we use GPy kernel, product can't be multiplied

    def kernel(self,stat1,stat2):
        return(self.state_kern(stat1[0:10],stat2[0:10])*self.action_kern(stat1[9],stat2[9]))

    def ret_dict(self):
        return self.state_dict

    def update_posterior(self,mean, covariance):
        return mean,covariance


    def cov_list(self,inp_state,inp_act):
        temp_list = np.array([])
        for _ in range(self.state_dict.shape[0]):
            temp_list=np.append(temp_list,[self.kernel(self.state_dict[_], np.append(inp_state,inp_act))])
        return temp_list.T

    def delta_fn(self):
        return self.delta_list
















