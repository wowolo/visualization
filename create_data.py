import numpy as np
from matplotlib import ticker, colors
import matplotlib.pyplot as plt

###############################################################################
############# create data
###############################################################################
d=2
d_out=32
n_samples=64
noise_scale=1/2
x_min=-1
x_max=1
n_val=256
np.random.seed(seed=3+2)

if d>1:
  x_train=np.random.normal(scale=1,size=(n_samples,d))
  x_train[:,0]=np.concatenate((np.array(np.linspace(x_min,x_max,int(n_samples/2))),np.array(np.linspace(x_min,x_max,int(n_samples/2)))))

  x_train[:,1]=np.concatenate((np.sin(1.5*np.pi*np.array(np.linspace(x_min,x_max,int(n_samples/2)))),-np.sin(1.5*np.pi*np.array(np.linspace(x_min,x_max,np.int(n_samples/2))))))
  x_val=np.random.normal(scale=1,size=(n_val,d))
else:
  x_train=np.array(np.linspace(x_min,x_max,n_samples)).reshape(n_samples,d)


#x_train[:,1]=0
y_min=-500.6
y_max=500.6

if d==1:
 if d_out==1:
  #f_true=lambda x_train : 2.0*(x_train>0)-1
  #f_true=lambda x_train : x_train[:,0]**2 -0.5
  #f_true=lambda x_train :2.0*(x_train[:,0]<0.3)*(x_train[:,0]-0.3) +1
  f_true=lambda x_train : np.sin(np.pi*x_train[:,0]) +0.0 
  #f_true=lambda x_train : np.sin(4.0*5*x_train[:,0])
 elif d_out==2:
  #f_true=lambda x_train : np.stack([np.sin(np.pi*x_train[:,0]) +0.0, 2.0*(x_train[:,0]<0)-1], axis=1)
  f_true=lambda x_train : np.stack([x_train[:,0]**2 -0.5, 2.0*(x_train[:,0]<0.3)*(x_train[:,0]-0.3)+1], axis=1)

 else:
  print('No f_true defined!')

 x_train = np.sort(x_train,0)
 resolution=1920
 x_smooth=np.array(np.linspace(x_min,x_max,resolution)).reshape(resolution,1)


else:
  resolution=(int(540),int(540))
  xx, yy = np.meshgrid(np.linspace(x_min,x_max,resolution[0]),np.linspace(y_min,y_max,resolution[1])) #for plotting

  #f_true=lambda x_train: x_train[:,0]**2+x_train[:,1]**2
  focusIndex=0
  def f_true(x):
    return np.array([1*x_train[:,1]**2-0.4,
                     #0.1*x_train[:,1]**2,
                     #0.1*x_train[:,1]**2,
                     #0.1*x_train[:,1]**2,
                     #0.1*x_train[:,1]**2,
                     0.4*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     np.exp(x_train[:,focusIndex])-1,
                     x_train[:,focusIndex],
                     x_train[:,focusIndex]**2-0.25,
                     x_train[:,focusIndex]**3,
                     -1*np.exp(-x_train[:,focusIndex])+0.5,
                     -3*x_train[:,focusIndex],
                     -3*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     -3*x_train[:,focusIndex]**3
                     -0.5*np.exp(x_train[:,focusIndex]-1),
                     -3*(x_train[:,focusIndex]-1),
                     -3*(x_train[:,focusIndex]-1)**2+1.5,
                     -3*(x_train[:,focusIndex]-1)**3,
                     1*np.exp(x_train[:,focusIndex]+1)-1,
                     2*(x_train[:,focusIndex]+1),
                     2*np.sign(x_train[:,focusIndex])*(x_train[:,focusIndex])**2,
                     2*(x_train[:,focusIndex]+1)**3,
                     0.2*np.exp(x_train[:,focusIndex]+0.5),
                     0.2*(x_train[:,focusIndex]+0.5),
                     0.2*(x_train[:,focusIndex]+0.5)**2,
                     0.2*(x_train[:,focusIndex]+0.5)**3,
                     -x_train[:,focusIndex]**2+0.5,
                     x_train[:,focusIndex]**2,
                     0.5*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     2*x_train[:,focusIndex]**3,
                     4*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     -np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     8*x_train[:,focusIndex],
                     256*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2,
                     1024*x_train[:,focusIndex],
                     64*x_train[:,focusIndex]**3,
                     -100*np.sign(x_train[:,focusIndex])*x_train[:,focusIndex]**2
                     ]).T
  #f_true=lambda x_train: x_train[:,0]**2+x_train[:,1]**2
#x_train[round(n_samples/2),:]=0
#y_train[round(n_samples/2)]=1


if d_out==1:
 #np.random.seed(seed=3+3)
 y_train=1.0*(   np.random.normal(scale=.1,size=n_samples)*noise_scale+ f_true(x_train)   ) #outputs training
 x_val=np.random.rand(n_val,d)*(x_max-x_min)+x_min
 y_val=f_true(x_val)
else:
 #np.random.seed(seed=3+3)
 y_train=1.0*(   np.random.normal(scale=0,size=(n_samples,d_out))*noise_scale+ f_true(x_train)   ) #outputs training
 y_val=f_true(x_val)


#x_train
#x_smooth
#plt.plot(x_train,y_train)
if d==1:
  plt.plot(x_train,y_train,'ko')
  plt.plot(x_val,y_val,'g.')

#np.stack([np.sin(np.pi*x_train[:,0]) +0.0, 2.0*(x_train[:,0]>0)-1], axis=1).shape
y_train.shape

if d_out>1:
  plt.plot(x_train[:,0],y_train[:,0],'g.')
  plt.plot(x_train[:,0],y_train[:,1],'r.')


###############################################################################
############# create data
###############################################################################