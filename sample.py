import numpy as np
from matplotlib import pyplot as plt
a=[2,3,4,5]
from scipy.stats import beta
new_belief=[1,2,4,8]
norm = sum(new_belief)
x=new_belief
import random
a=1
for i in range(50):
    a-=a*0.05
    if(i==25):
        print(a)

print(new_belief)
