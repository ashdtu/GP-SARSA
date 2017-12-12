
from matplotlib import pyplot as plt
import time
import numpy as np

from menu_model import SearchEnvironment
#from environments.continous_maze_discrete import CTS_Maze
from pomdp_task import SearchTask
#from tasks.CTS_TASK import CTS_MazeTask
from pybrain.rl.experiments import EpisodicExperiment
from learners.baseline_learner_menu import GP_SARSA
from agents.baseline_agent_menu import GPSARSA_Agent

for repeat in range(1):
    env = SearchEnvironment()  # goal
    #env=CTS_Maze((0.95,0.95))
    task = SearchTask(env,10)
    #task=CTS_MazeTask(env)
    learner = GP_SARSA(gamma=0.95)
    learner.sigma = 1
    learner.batchMode = False  # extra , not in use , set to True for batch learning
    agent = GPSARSA_Agent(learner)
    agent.logging = True

    exp = EpisodicExperiment(task, agent)
    agent.reset()
    sum=[]
    avg=[]
    performance=[]
    track_time=[]
    agent.init_exploration=1
    starttime = time.time()
    dict_size=[]
    epsilon=[]

    b=[]
    c=[]
    for num_exp in range(20):
        #print('new episode')
        performance=exp.doEpisodes(1)
        sum = np.append(sum, np.sum(performance))
        #if (num_exp % 50 == 0 and num_exp != 0):
        agent.init_exploration -= agent.init_exploration * 0.1
        #avg = np.mean(sum[num_exp-10:num_exp])
        print(np.sum(performance))
        #if(num_exp%10==0 and num_exp!=0):
        agent.learn()
        #print('reward',learner.ret_reward())
        agent.reset()

        #print(learner.state_dict.shape)
        #dict_size=np.append(dict_size,learner.state_dict.shape[0])
        track_time=np.append(track_time,[time.time()-starttime])
        #print(track_time)


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




















