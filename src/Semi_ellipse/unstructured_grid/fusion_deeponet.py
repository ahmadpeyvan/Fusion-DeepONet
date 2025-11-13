import jax
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import optax
import pickle
import jaxopt
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax import lax
from jax import nn

from jax.example_libraries import optimizers

num_epochs_adam = 50000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 28
BS_t = 8
variables = [0,1,2,3] #rho
npts = 10000
# lr = 1e-3
scaling = '01'

os.system('rm -r models')
os.system('mkdir models')

with open("training_datasetn.pkl", 'rb') as file:
    data_train = pickle.load(file)
    print("train=\t",len(data_train))

x_train  = data_train[0]
u_train  = data_train[1][:,variables]
v_train  = data_train[2]
np_train  = data_train[3]

with open("testing_datasetn.pkl", 'rb') as file:
    data_test = pickle.load(file)
    print("test=\t",len(data_test))

x_test  = data_test[0]
u_test  = data_test[1][:,variables]
v_test  = data_test[2]
np_test  = data_test[3]

print("v_train=\t",v_train.shape)
print("x_train=\t",x_train.shape)
print("u_train=\t",u_train.shape)
print("np_train=\t",np_train.shape)
print("v_test=\t",v_test.shape)
print("x_test=\t",x_test.shape)
print("u_test=\t",u_test.shape)
print("np_test=\t",np_test.shape)


Xmin = np.min(v_train)
Xmax = np.max(v_train)

dmin = np.zeros((1,len(variables)))
dmax = np.zeros((1,len(variables)))
fac  = 1e-10*np.ones_like(dmin)

for i in range(len(variables)):
    dmin[0,i] = np.min(u_train[:,i])
    dmax[0,i] = np.max(u_train[:,i])

oness = np.ones((1,len(variables)))
if scaling=='01':
    u_train = (u_train - dmin)/(dmax - dmin)
    u_test = (u_test- dmin)/(dmax - dmin)
    tol = (fac-dmin)/(dmax - dmin)
    v_train = (v_train - Xmin)/(Xmax - Xmin) 
    v_test = (v_test- Xmin)/(Xmax - Xmin) 
else:
    u_train = 2*(u_train - dmin)/(dmax - dmin) - oness
    u_test = 2*(u_test- dmin)/(dmax - dmin) - oness
    tol = 2*(fac-dmin)/(dmax - dmin) - oness
    v_train = 2.*(v_train - Xmin)/(Xmax - Xmin) - 1.0
    v_test = 2.*(v_test- Xmin)/(Xmax - Xmin) - 1.0


maxn = np.max(np_train)
indm = np.argmax(np_train)

u_tr = np.zeros((BS,maxn,len(variables)))
u_te = np.zeros((BS_t,maxn,len(variables)))
x_tr = np.zeros((BS,maxn,2))
x_te = np.zeros((BS_t,maxn,2))
print("maxn=\t",maxn,indm)
print("u_t=\t",u_tr.shape)
print("u_t=\t",u_te.shape)

# u_last = u_train[]
# repeating last entry of trajectories to get to similar shapes across samples
j1 = 0
j2 = 0
for i in range(BS):
    j2 = j2+np_train[i]
    # print("j1,j2=\t",j1,j2)
    u_tr[i,j1-j1:j2-j1,:]=u_train[j1:j2,:]
    x_tr[i,j1-j1:j2-j1,:]=x_train[j1:j2,:]
    vv = u_train[j2-1,:]
    vv = vv[None,:]
    xx = x_train[j2-1,:]
    xx = xx[None,:]
    print("j2-j1=\t",maxn-(j2-j1))
    print("shape=\t",u_tr.shape)
    temp = np.tile(vv, (maxn-(j2-j1), 1))
    temp1 = np.tile(xx, (maxn-(j2-j1), 1))
    u_tr[i,j2-j1:,:]=temp
    x_tr[i,j2-j1:,:]=temp1
    j1 += np_train[i] 

print("u_tr=\t",u_tr.shape)
print("x_tr=\t",x_tr.shape)

j1 = 0
j2 = 0
for i in range(BS_t):
    j2 = j2+np_test[i]
    # print("j1,j2=\t",j1,j2)
    u_te[i,j1-j1:j2-j1,:]=u_test[j1:j2,:]
    x_te[i,j1-j1:j2-j1,:]=x_test[j1:j2,:]
    vv = u_test[j2-1,:]
    vv = vv[None,:]
    xx = x_test[j2-1,:]
    xx = xx[None,:]
    temp = np.tile(vv, (maxn-(j2-j1), 1))
    temp1 = np.tile(xx, (maxn-(j2-j1), 1))
    u_te[i,j2-j1:,:]=temp
    x_te[i,j2-j1:,:]=temp1
    j1 += np_test[i] 

print("u_te=\t",u_te.shape)
print("x_te=\t",x_te.shape)



def save_model(param,n):
    filename = './models/model'+'.'+str(n)
    with open(filename, 'wb') as file:
        pickle.dump(param, file)

initializer = jax.nn.initializers.glorot_normal()

def hyper_initial_WB(layers,key):
    L = len(layers)
    W = []
    b = []
    for l in range(1, L):
        in_dim = layers[l-1]
        out_dim = layers[l]
        std = np.sqrt(2.0/(in_dim+out_dim))
        weight = initializer(key, (in_dim, out_dim), jnp.float32)*std
        bias = initializer(key, (1, out_dim), jnp.float32)*std
        W.append(weight)
        b.append(bias)
    return W, b

def hyper_parameters_A(shape):
    return jnp.full(shape, 0.1, dtype=jnp.float32)

def hyper_parameters_amplitude(shape):
    return jnp.full(shape, 0.0, dtype=jnp.float32)

def hyper_parameters_freq1(shape):
    return jnp.full(shape, 0.1, dtype=jnp.float32)

 

def hyper_initial_frequencies(layers):

    L = len(layers)

    a = []
    c = []

    a1 = []
    F1 = []
    c1 = []
    
    for l in range(1, L):

        a.append(hyper_parameters_A([1]))
        c.append(hyper_parameters_A([1]))

        a1.append(hyper_parameters_amplitude([1]))
        F1.append(hyper_parameters_freq1([1]))
        c1.append(hyper_parameters_amplitude([1]))

    return a, c, a1, F1, c1 

def shuffle_in_minibatch(v,x,u):
    # print("in=\t",u.shape)
    ind = np.arange(u.shape[0])
    # print("ind=\t",ind)
    np.random.shuffle(ind)
    u = u[ind]
    v = v[ind]
    x = x[ind]
    return jnp.array(v),jnp.array(x),jnp.array(u)

def fnn_fuse_mixed_add(Xt, Xb, pt,pb):
    Wt, bt, at, ct, a1t, F1t, c1t = pt
    Wb, bb, ab, cb, a1b, F1b, c1b = pb

    inputst = Xt
    inputsb = Xb

    skip = []
    L = len(Wb)
    for i in range(L-1):
        inputsb =  jnp.tanh(jnp.add(10*ab[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),cb[i])) \
            + 10*a1b[i]*jnp.sin(jnp.add(10*F1b[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),c1b[i]))
        skip.append(inputsb)
    for i in range(1,L-1):
        skip[i] = jnp.add(skip[i],skip[i-1])

    for i in range(L-1):
        # print("inputst1=\t",inputst.shape)
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
        inputst = jnp.multiply(inputst,skip[i][:, None,:])
        
    Yt = jnp.dot(inputst, Wt[-1]) + bt[-1]     
    Yb = jnp.dot(inputsb, Wb[-1]) + bb[-1]

    return Yt, Yb

# #input dimension for Branch Net
u_dim = v_train.shape[-1]

#output dimension for Branch and Trunk Net
G_dim = 64

#Branch Net
layers_f = [u_dim] + [64]*3 + [len(variables)*G_dim]
# print("branch layers:\t",layers_f)

# Trunk dim
x_dim = 2

#Trunk Net
layers_x = [x_dim] + [64]*3 + [G_dim]
# print("Trunks layers:\t",layers_x)
key = random.PRNGKey(1234)
# key = random.PRNGKey(6534)
key1, key2 =  random.split(key, num=2)

W_branch, b_branch = hyper_initial_WB(layers_f,key1)
a_branch, c_branch, a1_branch, F1_branch , c1_branch = hyper_initial_frequencies(layers_f)
# key1 = random.PRNGKey(6534)
W_trunk, b_trunk = hyper_initial_WB(layers_x,key2)
a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk = hyper_initial_frequencies(layers_x)

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params
    v, x = data

    u_out_trunk,u_out_branch = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)

    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)  

    return u_pred

def loss(params, data, u):
    u_preds = predict(params, data)
    loss_data = jnp.mean((u_preds.flatten() - u.flatten())**2)
    mse = loss_data

    return mse

@jit
def update(params, data, u, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss,argnums=0,has_aux=False)(params, data, u)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

learning_rate = optimizers.exponential_decay(0.001,2000,0.91)
# opt_init, opt_update, get_params = Lion(lr=learning_rate)
opt_init, opt_update, get_params = optimizers.adam(learning_rate)

opt_state=opt_init([W_branch,b_branch,W_trunk, b_trunk,
    a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk,
    a_branch, c_branch, a1_branch, F1_branch , c1_branch])

params = get_params(opt_state)
train_loss, test_loss = [], []

v_tr = jnp.array(v_train)
v_te = jnp.array(v_test)
u_tr , x_tr = jnp.array(u_tr),jnp.array(x_tr)
u_te , x_te = jnp.array(u_te),jnp.array(x_te)
epo = []
start_time = time.time()
n = 0

start_time_tot = time.time()
for epoch in range(num_epochs_adam):
    epoch_time = time.time()

    params, opt_state, loss_val = update(params, [v_tr,x_tr], u_tr, opt_state)
    losss = loss_val
    # print(epoch,losss)
    if epoch % 10 ==0:
        epoch_time = time.time() - start_time
        loss_test= loss(params, [v_te, x_te],u_te)
        u_train_pred = predict(params, [v_tr, x_tr])
        u_train_pred = u_train_pred.flatten()
        u_i = u_tr.flatten()
        l2_train = jnp.mean(jnp.linalg.norm(u_i - u_train_pred, 2)/jnp.linalg.norm(u_i , 2))

        u_test_pred = predict(params, [v_te, x_te])
        u_test_pred = u_test_pred.flatten()
        u_i = u_te.flatten()
        l2_test = jnp.mean(jnp.linalg.norm(u_i - u_test_pred, 2)/jnp.linalg.norm(u_i , 2))

        train_loss.append(losss)
        test_loss.append(loss_test)
        epo.append(epoch)
        print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} | Test MSE: {:0.3e} | Train L2: {:0.5f} | Test L2: {:0.5f}".format(epoch, epoch_time,losss,loss_test,l2_train, l2_test))

    if epoch % 20 ==0:
        save_model(params,n)
        filename = './models/loss'
        with open(filename, 'wb') as file:
            pickle.dump((epo,train_loss,test_loss), file)
        n += 1
            
        # train_loss.append(l1)
        
    start_time = time.time()
    
time_tot = time.time() - start_time_tot
print("total time=\t",time_tot)

save_model(params,n)

filename = './models/loss'
with open(filename, 'wb') as file:
    pickle.dump((epo,train_loss,test_loss), file)

