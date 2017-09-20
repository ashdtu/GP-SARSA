temp_list=[]
import numpy as np
with open("menu_reward.txt") as file:
    for line in file:
        line = line.strip()
        line = float(line)
        temp_list.append(line)

avg = [np.mean(temp_list[i:i + 100]) for i in np.arange(0,10000,100)]
from matplotlib import pyplot as plt
plt.plot(range(100),avg)
plt.title("Mean Reward per 100 trials")
#plt.show()

a=np.array([10,50,70,-30,-80])
print(a.any()   <0)
for some in a:
    if(some<0):
        print('eror')