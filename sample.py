import numpy as np
a=[2,3,4,5]
from scipy.stats import beta
new_belief=[1,2,4,8]
norm = sum(new_belief)
x=new_belief

a=1.0
for i in range(100):
    a-=a*0.05
    print(a)

print(a)
print(new_belief)