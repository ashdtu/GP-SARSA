import numpy as np
from scipy.stats import beta
new_belief=[1,2,4,8]
norm = sum(new_belief)
new_belief = (new_belief/norm)
print(new_belief)