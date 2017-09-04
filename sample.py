import numpy as np
a=[2,3,4,5]
from scipy.stats import beta
new_belief=[1,2,4,8]
norm = sum(new_belief)

print(new_belief)
print('sub',np.subtract(a,new_belief)**2)
a=[(0,3),(4,5)]
print(a[1][1])
print(np.array(new_belief)/10)