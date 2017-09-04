

from pybrain.rl.agents.logging import LoggingAgent
import random
from menu_model import SearchEnvironment
import numpy as np

class GPSARSA_Agent(LoggingAgent):

    def __init__(self, learner, **kwargs):
        LoggingAgent.__init__(self, 9, 1, **kwargs)
        self.learner = learner
        self.reset()
        self.learning=True
        self.learner.dataset=self.history
        self.visited_states_x=[]
        self.visited_states_y=[]
        self.qvalues=[]
        self.actionvalues=[]
        self.init_exploration=1.0

    def _actionProbs(self, state):


        self.q_mean=[]
        self.q_cov=[]
        i=0
        for act in SearchEnvironment.actions :
            self.K=[]
            for i in range(self.learner.ret_dict().shape[0]):

                self.K=np.append(self.K,self.learner.kernel(self.learner.ret_dict()[i],np.append(state,act))) #k(s,a) with previous sequence


            alpha=self.learner.u_tilde
            C=self.learner.C_tilde
            self.q_mean=np.append(self.q_mean,np.dot(self.K,alpha)) #q mean list for every action
            self.q_cov=np.append(self.q_cov,self.learner.kernel(np.append(state,act),np.append(state,act))-np.dot(np.dot(self.K,C),self.K.T)) #q_covariance for every action


        return self.q_mean,self.q_cov



    def getAction(self):
        action=None
        if (self.learner.ret_dict() is not None):
            q_meanlist,q_covlist = self._actionProbs(self.lastobs)
            #print('in agent',self.lastobs)
            if (random.random() > self.init_exploration):
                action = SearchEnvironment.actions[np.argmax(q_meanlist)]
                #self.actionvalues.append(action)
            else:
                action = random.choice(SearchEnvironment.actions)
                #action=SearchEnvironment.actions[np.argmax(q_covlist)]

        else:
            action=random.choice(SearchEnvironment.actions)

        self.lastaction = action

        return action

    def integrateObservation(self, obs):
        LoggingAgent.integrateObservation(self, obs)

    def reset(self):
        LoggingAgent.reset(self)        #clear dataset sequences
        self.learner.reset()
        self.newEpisode()

    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            self.history.newSequence()


    def learn(self):
        if not self.learning:
            return
        self.learner.learn()


    def ret_values(self):
        return self.actionvalues,self.qvalues

