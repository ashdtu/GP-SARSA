
from matplotlib import pyplot as plt
import time
import numpy as np

from menu_model import SearchEnvironment
from pomdp_task import SearchTask
from pybrain.rl.experiments import EpisodicExperiment
from learners.sparse_learner import GP_SARSA_SPARSE
from agents.sparse_agent import GPSARSA_Agent

plt.ion()

i=1000
performance=[]  #reward accumulation, dump variable for any evaluation metric
sum=[]


track_time=[]
dict_size=[]

for repeat in range(1):
    env = SearchEnvironment()  # goal

    task = SearchTask(env,20)
    learner = GP_SARSA_SPARSE(gamma=0.95)
    learner.sigma = 1
    learner.batchMode = False  # extra , not in use , set to True for batch learning
    agent = GPSARSA_Agent(learner)
    agent.logging = True

    exp = EpisodicExperiment(task, agent)
    agent.reset()
    sum=[]
    performance=[]
    #track_time=[]
    agent.init_exploration=1.0
    #starttime = time.time()
    dict_size=[]
    epsilon=[]

    b=[]
    c=[]
    for num_exp in range(100):
        print('new episode')
        performance=exp.doEpisodes(1)
        sum = np.append(sum, np.sum(performance))
        #for a,b,c in agent.history:
        #    print('state',a)
        #    print('action',b)
        #    print('reward',c)

        agent.init_exploration=(10/(10+num_exp))


        #epsilon.append(agent.init_exploration)
        agent.learn()
        #print('state_Dict',learner.state_dict)


        #print('dataset',agent.history)

        #dict_size=np.append(dict_size,learner.state_dict.shape[0])
        #track_time=np.append(track_time,[time.time()-starttime])

        agent.reset()
        print(sum)




