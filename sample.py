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
plt.show()