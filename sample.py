temp_list=[]
import numpy as np
with open("menu_reward.txt") as file:
    for line in file:
        line = line.strip()
        line = float(line)
        temp_list.append(line)

#avg = [np.mean(temp_list[i:i + 100]) for i in np.arange(0,10000,100)]
from matplotlib import pyplot as plt
#plt.plot(range(100),avg)
#plt.title("Mean Reward per 100 trials")
#plt.show()

a=np.array([10,50,70,-30,-80])
a[a<10]=0
print(a)
'''
print(a.any()   <0)
for some in a:
    if(some<0):
        print('eror')

a=[0.11,0.11]
def sample(inp):
    #check=inp.copy()
    if inp[0]==0.11:
        inp[0]=0.2*0.11

sample(a)
print(a)
'''
c=np.array([4])
d=np.array([5])
c=c[:,np.newaxis]
f=np.hstack((c,d[:,np.newaxis]))
e=np.array([3])
g=np.vstack((c,e[:,np.newaxis]))
print(np.dot(f,g))
seq=[([4,5],[5],[67]),([6,5],[6],[7])]

some=np.array([1,1,2,4])
other=np.array([1,1,2,6])

print(np.dot(some,other))

abc=np.ones((3,3))
x=np.ones(3)

lol=np.dot(abc,x[:,np.newaxis])
lol=np.reshape(lol,3)
print('bro',lol+np.ones(3))


