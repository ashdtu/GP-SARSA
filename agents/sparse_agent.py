

from pybrain.rl.agents.logging import LoggingAgent
import random
from pybrain.rl.environments.continous_maze_discrete import CTS_Maze
import numpy as np

class GPSARSA_Agent(LoggingAgent):


    init_exploration = 0.30  # aka epsilon
    exploration_decay = 0.10  # per episode


    def __init__(self, learner, **kwargs):
        LoggingAgent.__init__(self, learner.num_features, 1, **kwargs)
        self.learner = learner
        #self.reset()
        self.learning=True
        self.learner.dataset=self.history


    def _actionProbs(self, state):


        self.q_mean=[]
        self.q_cov=[]
        i=0
        for act in CTS_Maze.actions :
            self.K=[]
            for i in range(self.learner.ret_dict().shape[0]):

                self.K=np.append(self.K,self.learner.kernel(self.learner.ret_dict()[i],np.append(state,act))) #k(s,a) with previous sequence

            #cum_reward=self.learner.ret_reward()
            alpha=learner.u_tilde
            C=learner.C_tilde
            self.q_mean=np.append(self.q_mean,np.dot(self.K,alpha)) #q mean list for every action
            #self.q_cov=np.append(self.q_cov,self.learner.kernel(np.append(state,act),np.append(state,act))-np.dot(np.dot(self.K,C),self.K.T)) #q_covariance for every action

        #print(self.q_mean[0])
        #print(self.q_cov[0])
        return self.q_mean,self.q_cov





    def getAction(self):
        action=None
        if (self.learner.ret_dict() is not None):
            q_meanlist, q_covlist = self._actionProbs(self.lastobs)
            if (random.random() > self.init_exploration):
                action = CTS_Maze.actions[np.argmax(q_meanlist)]


            else:
                action = random.choice(CTS_Maze.actions)
                #action=CTS_Maze.actions[np.argmax(q_covlist)]
        else:
            action=random.choice(CTS_Maze.actions)

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


