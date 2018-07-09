import numpy as np
from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

class CTS_Maze(Environment, Named):
	
    indim=2
    outdim=1
    discreteStates = False
    discreteActions = True
    #numActions = 8
    # current state
    perseus=np.array([0,0],dtype=float)
    # default initial state
    initPos = np.array([0,0],dtype=float)
    timesteps=0

    actions=[0,45,90,135,180,225,270,315]


    def __init__(self,goal):
        self.goal=np.array(goal,dtype=float)
        self.n = 0
        self.initPos=np.array([0.0,0.0],dtype=float)
        #print(self.initPos)
        self.perseus=self.initPos


    def obstacle_fn(self,inp):

        check=inp
        if(inp[0]<0 or inp[0]>1 or inp[1]<0 or inp[1]>1):
            return True
        elif(0.60<=check[0]<=0.85 and 0.60<=check[1]<=0.85):
            return True
        else:
            return False

    def take_action(self,state,action):
        #+np.random.uniform(-30,30) #stochasticity of +/- 30 degrees
        new_state=np.array([0,0],dtype=float)
        new_state[0]=state[0]+0.1*np.cos(np.deg2rad(action))
        new_state[1]=state[1]+0.1*np.sin(np.deg2rad(action))
        #return np.around(new_state,decimals=2)
        return new_state

    def performAction(self, action):
        temp=self.take_action(self.perseus,action)
        if( not self.obstacle_fn(np.around(temp,decimals=2))):
            self.perseus=temp
        else:
            self.perseus=self.perseus
        self.timesteps+=1

    def getSensors(self):
        #return (np.around(self.perseus,decimals=2))
        return self.perseus

    def reset(self):
        self.perseus=self.initPos
        self.timesteps=0

