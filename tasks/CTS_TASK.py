
from pybrain.rl.environments import Task
import numpy as np
class CTS_MazeTask(Task):
    """ This is a MDP task for the CTS MazeEnvironment. The state is fully observable,
        giving the agent the current position of perseus. Reward is given on reaching
        the goal, otherwise a NEGATIVE reward. """

    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        if (self.env.goal[0]-0.05<=self.env.perseus[0]<=self.env.goal[0]+0.05 and self.env.goal[1]-0.05<=self.env.perseus[1]<=self.env.goal[1]+0.05):

            reward = 10000
        else:
            reward = -400
        return reward

    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given
            to the super class.
        """
        Task.performAction(self, action)


    def getObservation(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        obs = self.env.perseus
        return obs

    def reset(self):

        self.env.reset()


    def isFinished(self):
        self.env.goal=np.array(self.env.goal,dtype=float)
        return ((self.env.goal[0]-0.05<=self.env.perseus[0]<=self.env.goal[0]+0.05 and self.env.goal[1]-0.1<=self.env.perseus[1]<=self.env.goal[1]+0.05) or self.env.timesteps==40)



