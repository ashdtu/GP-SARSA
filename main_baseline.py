
from matplotlib import pyplot as plt

import numpy as np

from environments.continous_maze_discrete import CTS_Maze
from tasks.CTS_TASK import CTS_MazeTask
from pybrain.rl.experiments import EpisodicExperiment
from learners.baseline_learner import GP_SARSA
from agents.baseline_agent import GPSARSA_Agent
env=CTS_Maze([0.40,0.40]) #goal

task=CTS_MazeTask(env)
learner=GP_SARSA(gamma=0.95)
learner.batchMode=False #extra , not in use , set to True for batch learning
agent=GPSARSA_Agent(learner)
agent.logging=True

exp=EpisodicExperiment(task,agent) #epsilon greedy exploration (with and without use of uncertainity)
plt.ion()

i=1000
performance=[]  #reward accumulation, dump variable for any evaluation metric
sum=[]
agent.reset()
i=0
for num_exp in range(100):

    performance=exp.doEpisodes(1)
    sum = np.append(sum, np.sum(performance))

    if(num_exp%10==0):
        agent.init_exploration-=agent.init_exploration*0.10

    agent.learn()
    print('alpha',np.dot(learner.inv,learner.ret_reward().T))

    agent.reset()

'''
b=learner.ret_cov()

a=learner.state_dict

print(sum)
epis=range(num_exp+1)
plt.plot(epis,sum)
plt.pause(0.5)

rewardfile=open('reward_300_full_cov_fixed.txt','w')
for some in sum:
    rewardfile.write("%s \n" %some)
thefile=open('state_dic.txt','w')
cov_file=open('covar.txt','w')
for item in range(a.shape[0]):
    thefile.write("%s\n" %a[item])
    cov_file.write("%s\n" %b[item])
'''




















