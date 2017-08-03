
from matplotlib import pyplot as plt

import numpy as np

from pybrain.rl.environments.continous_maze_discrete_fixed import CTS_Maze
from pybrain.rl.environments.CTS_TASK import CTS_MazeTask
from pybrain.rl.experiments import EpisodicExperiment
from gpsarsa_sparse import GP_SARSA_SPARSE
from pybrain.rl.agents.policy_agent import GPSARSA_Agent
env=CTS_Maze([0.50,0.50]) #goal

task=CTS_MazeTask(env)
learner=GP_SARSA_SPARSE(gamma=0.95)
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
for num_exp in range(1):

    performance=exp.doEpisodes(1)
    sum = np.append(sum, np.sum(performance))

    if(num_exp%30==0):
        agent.init_exploration-=agent.init_exploration*agent.exploration_decay

    agent.learn()

    #print(learner.ret_reward().shape)
    #print(performance)
    #print(learner.state_dict)

    #print(learner.ret_dict())
    #print(learner.ret_h().shape)
    #print(learner.ret_cov())
    #print(learner.covariance_list.shape)
    #print(learner.state_dict)
    print(np.sum(performance))

    agent.reset()


b=learner.ret_cov()
#print(b)
a=learner.state_dict
#print(learner.ret_reward().shape)
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







    #agent.learn()
    #plt.plot(num_exp,np.sum(performance),c='r')
    #plt.show()
    #plt.pause(0.05)

    #print(agent.getAction())#do one epsiode, agent stores in logging agent, uses previous Q posterior estimate
    #agent.learn()

    #agent.reset()#update H and k and return INV for alpha
    #agent.reset() #history cleared
    #plt.scatter(env.getSensors()[0], env.getSensors()[1])
    #plt.pause(0.005)
    #plt.pause(0.005)
'''various evaluation criterias'''

    #plt.plot(agent.getAction())
    #plt.pause(0.05)
    #exp.agent=testagent
    #r=np.mean([sum(x) for x in exp.doEpisodes(5)])

    #testagent.reset()
    #exp.agent=agent
    #performance.append(r)
    #plt.plot(r,env.timesteps)
#Test for dataset historyt
'''for i in range(2):
    for state,action,reward in agent.history.getSequenceIterator(i):
        print(state,action,reward)
'''
















