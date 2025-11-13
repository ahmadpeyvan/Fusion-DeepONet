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

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Defining an optimizer in Jax
num_epochs_adam = 100000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 48
BS_t = 12
variables = [4] #temperature
npts = 10000
eps = 1e-8
# lr = 1e-3
scaling = '01'

os.system('rm -r models_discrete')
os.system('mkdir models_discrete')

data_train = np.load("training_dataset.npz")
data_test = np.load("testing_dataset.npz")


v_train = data_train["v_train"]
x_train = data_train["x_train"][:,:,0:2]
u_train = data_train["u_train"][:,:,variables]

v_test = data_test["v_test"]
x_test = data_test["x_test"][:,:,0:2]
u_test = data_test["u_test"][:,:,variables]


# np_train = np_train.flatten()[0:,None]
# np_test = np_test.flatten()[0:,None]
print("v_train=\t",v_train.shape)
print("x_train=\t",x_train.shape)
print("u_train=\t",u_train.shape)

print("v_test=\t",v_test.shape)
print("x_test=\t",x_test.shape)
print("u_test=\t",u_test.shape)


Xmin = np.min(v_train)
Xmax = np.max(v_train)

dmin = np.zeros((1,1,len(variables)))
dmax = np.zeros((1,1,len(variables)))

fac  = 1e-10*np.ones_like(dmin)

for i in range(len(variables)):
    dmin[0,0,i] = np.min(u_train[:,:,i])
    dmax[0,0,i] = np.max(u_train[:,:,i])
print("dmin=\t",dmin)
print("dmax=\t",dmax)
oness = np.ones((1,1,len(variables)))
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

def save_model(param,n):
    filename = './models_discrete/model'+'.'+str(n)
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
    # inputs = X#2.*(X - Xmin)/(Xmax - Xmin) - 1.0
    # print("T first input=\t",inputs.shape)
    inputst = Xt
    inputsb = Xb

    # print("inputst=\t",inputst.shape)
    # print("inputsb=\t",inputsb.shape)

    skip = []
    L = len(Wb)
    for i in range(L-1):
        inputsb =  jnp.tanh(jnp.add(10*ab[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),cb[i])) \
            + 10*a1b[i]*jnp.sin(jnp.add(10*F1b[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),c1b[i]))
        # print("inputsb=\t",inputsb.shape)
        skip.append(inputsb)
    for i in range(1,L-1):
        skip[i] = jnp.add(skip[i],skip[i-1])
    # print("inputst0=\t",inputst.shape)

    for i in range(L-1):
        # print("inputst1=\t",inputst.shape)
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
        inputst = jnp.multiply(inputst,skip[i][:, None,:])
        # inputst = jnp.multiply(inputst,skip_time[i][None,:, None,:])
        
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

    # print("v=\t",v.shape)
    # print("x=\t",x.shape)

    u_out_trunk,u_out_branch = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    # print("u_out_trunk=\t",u_out_trunk.shape)
    # print("u_out_branch1=\t",u_out_branch.shape)
    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)
    # print("u_out_branch2=\t",u_out_branch.shape)
    # print("u_out_trunk2=\t",u_out_trunk.shape)
    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)  
    # print("u_pred=\t",u_pred.shape)
    return u_pred

def loss(params, data, u,deriv_data):
    u_preds = predict(params, data)
    v, x = data
    diff_coord, diff_u = deriv_data

    diff_pred =jnp.diff(u_preds,axis=1)

    der1 = (diff_u - diff_pred)/diff_coord

    loss = jnp.mean((u_preds.flatten() - u.flatten())**2)
    loss1 = jnp.mean((der1.flatten())**2)

    mse = loss+1e-3*loss1
    return mse

def loss_for_test(params, data, u, deriv_data):
    u_preds = predict(params, data)
    v, x = data
    diff_coord, diff_u = deriv_data

    diff_pred =jnp.diff(u_preds,axis=1)

    der1 = (diff_u - diff_pred)/diff_coord

    loss = jnp.mean((u_preds.flatten() - u.flatten())**2)
    loss1 = jnp.mean((der1.flatten())**2)

    denom = diff_u/diff_coord

    l2 = jnp.mean(jnp.linalg.norm(u.flatten() - u_preds.flatten(), 2)/jnp.linalg.norm(u.flatten() , 2))
    l2_diff = jnp.mean(jnp.linalg.norm(der1.flatten(), 2)/jnp.linalg.norm(denom.flatten() , 2))

    mse = loss+1e-3*loss1
    return mse, loss, loss1, l2, l2_diff

@jit
def update(params, data, u,deriv_data, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss,argnums=0,has_aux=False)(params, data, u,deriv_data)
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
# v_i = jnp.array(v_test)
# u_i,x_i = jnp.array(u_te),jnp.array(x_te) #sample_points(u_train[i],x_train[i],npts)
# u_train_pred = predict(params, [v_i, x_i])
v_tr = jnp.array(v_train)
v_te = jnp.array(v_test)
u_tr , x_tr = jnp.array(u_train),jnp.array(x_train)
u_te , x_te = jnp.array(u_test),jnp.array(x_test)
epo = []

diff_train = jnp.diff(u_tr,axis=1)
diff_test = jnp.diff(u_te,axis=1)

diff_coord =jnp.diff(x_train,axis=1)
print("diff_coord1=\t",diff_coord.shape)
diff_coord = jnp.linalg.norm(diff_coord, axis=2, keepdims=True)
print("diff_coord2=\t",diff_coord.shape)
diff_coord_train = jnp.where(diff_coord <= eps, 1.0, diff_coord)
print("diff_coord_train=\t",diff_coord_train.shape)

diff_coord =jnp.diff(x_test,axis=1)
diff_coord = jnp.linalg.norm(diff_coord, axis=2, keepdims=True)
diff_coord_test = jnp.where(diff_coord <= eps, 1.0, diff_coord)

deriv_data_train = [diff_coord_train,diff_train]
deriv_data_test = [diff_coord_test,diff_test]

start_time = time.time()
n = 0
# loss_train= loss(params, [v_tr, x_tr],u_tr)

# u_train_pred = predict(params, [v_tr,x_tr])
start_time_tot = time.time()
for epoch in range(num_epochs_adam):
    epoch_time = time.time()
    # v_tr,x_tr,u_tr = shuffle_in_minibatch(v_train,x_tr,u_tr)
    params, opt_state, loss_val = update(params, [v_tr,x_tr], u_tr,deriv_data_train, opt_state)
    # print(epoch,losss)
    if epoch % 10 ==0:
        epoch_time = time.time() - start_time

        losss,l_train,l_der_train,l2_train,l2_diff_train = loss_for_test(params,[v_tr,x_tr], u_tr,deriv_data_train)
        loss_test,l_test,l_der_test,l2_test,l2_diff_test = loss_for_test(params,[v_te,x_te], u_te,deriv_data_test)

        train_loss.append(losss)
        test_loss.append(loss_test)
        epo.append(epoch)
        print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} | MSE: {:0.3e}| MSE_der: {:0.3e} | Test MSE: {:0.3e} | Train L2: {:0.5f} | Test L2: {:0.5f} | Diff_Train L2: {:0.5f} | Diff_Test L2: {:0.5f}".format(epoch, epoch_time,losss,l_train,l_der_train,loss_test,l2_train, l2_test,l2_diff_train,l2_diff_test))

    if epoch % 20 ==0:
        save_model(params,n)
        filename = './models_discrete/loss'
        with open(filename, 'wb') as file:
            pickle.dump((epo,train_loss,test_loss), file)
        n += 1
            
        
    start_time = time.time()
    
time_tot = time.time() - start_time_tot
print("total time=\t",time_tot)

save_model(params,n)

filename = './models_discrete/loss'
with open(filename, 'wb') as file:
    pickle.dump((epo,train_loss,test_loss), file)

