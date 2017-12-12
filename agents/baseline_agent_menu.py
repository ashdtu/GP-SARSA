

from pybrain.rl.agents.logging import LoggingAgent
import random
from menu_model import SearchEnvironment
#from menu_model_short import SearchEnvironment
import numpy as np

class GPSARSA_Agent(LoggingAgent):


    init_exploration = 1.0  # aka epsilon
    #exploration_decay = 0.10  # per episode


    def __init__(self, learner, **kwargs):
        LoggingAgent.__init__(self,9,1, **kwargs)
        self.learner = learner
        #self.reset()
        self.learning=True
        self.learner.dataset=self.history

    def _actionProbs(self, state):
        self.q_mean=[]
        self.q_cov=[]
        i=0
        for act in SearchEnvironment.actions :
            self.K=[]
            for i in range(self.learner.ret_dict().shape[0]):

                self.K=np.append(self.K,self.learner.kernel(self.learner.ret_dict()[i],np.append(state,act))) #k(s,a) with previous sequence
            self.K=np.reshape(self.K,(1,len(self.K)))
            cum_reward=np.reshape(self.learner.ret_reward(),(len(self.learner.ret_reward()),1))
            #print('inv dot',np.dot(self.learner.inv,cum_reward))
            self.q_mean=np.append(self.q_mean,np.dot(self.K,np.dot(self.learner.inv,cum_reward))) #q mean list for every action
            self.q_cov=np.append(self.q_cov,self.learner.kernel(np.append(state,act),np.append(state,act))-np.dot(np.dot(self.K,self.learner.inv),np.dot(self.learner.ret_h(),self.K.T))) #q_covariance for every action

        print(max(self.q_mean),np.argmax(self.q_mean))
        print(min(self.q_mean),np.argmin(self.q_mean))

        return self.q_mean,self.q_cov

    def getAction(self):
        action=None
        if (self.learner.ret_dict() is not None):
            q_meanlist, q_covlist = self._actionProbs(self.lastobs)
            #print('mean',q_meanlist)
            for some in q_meanlist:
                if some>10000 or some<-16000:
                    print('error',q_meanlist)
                    #raise ValueError('wtf')
            if (random.random() > self.init_exploration):
                action = SearchEnvironment.actions[np.argmax(q_meanlist)]

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
        LoggingAgent.reset(self) #clear dataset sequences

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


