
from matplotlib import pyplot as plt
import time
import numpy as np

from menu_model import SearchEnvironment
from pomdp_task import SearchTask
from pybrain.rl.experiments import EpisodicExperiment
from learners.sparse_learner import GP_SARSA_SPARSE
from agents.sparse_agent import GPSARSA_Agent

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
    avg=[]
    performance=[]
    track_time=[]
    agent.init_exploration=1.0
    #starttime = time.time()
    dict_size=[]
    epsilon=[]

    b=[]
    c=[]
    for num_exp in range(30000):
        #print('new episode')
        performance=exp.doEpisodes(1)
        if(num_exp%1000==0 and num_exp!=0):
            agent.init_exploration-=agent.init_exploration*0.1
        sum=np.append(sum,np.sum(performance))
        agent.learn()
        print(sum)
        #print(learner.state_dict.shape)
        #dict_size=np.append(dict_size,learner.state_dict.shape[0])
        #track_time=np.append(track_time,[time.time()-starttime])
        agent.reset()
    avg=[np.mean(sum[i:i+1000]) for i in np.arange(0,30000,1000)]

    print(avg)




