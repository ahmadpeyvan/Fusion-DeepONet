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
from scipy.spatial import cKDTree

from jax.example_libraries import optimizers

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Defining an optimizer in Jax
num_epochs_adam = 100000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 42
BS_t = 18
variables = [0,1,2,3,4] #rho
eps = 1e-8
npts = 10000
normal_y = jnp.array([0.0, 1.0])
normal_x = jnp.array([1.0, 0.0])
# lr = 1e-3
scaling = '01'

filename_loss = './models/loss'

os.system('rm -r models')
os.system('mkdir models')

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

def shuffle_in_points(v,x,u):
    # print("in=\t",u.shape)
    ind = np.arange(u.shape[1])
    # print("ind=\t",ind)
    np.random.shuffle(ind)
    u = u[:,ind,:]
    # v = v[ind]
    x = x[:,ind,:]
    return jnp.array(v),jnp.array(x),jnp.array(u)


def compute_nn_idx(coords, k):
    # Input coords: (Npts, 2) as a NumPy array.
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    return indices[:, 1:]  # discard self-index

def directional_derivative_single_case_jax(coords, values, v, nn_idx, k=5):
    # coords: (Npts, 2), values: (Npts,), v: (2,), nn_idx: (Npts, k)
    values = values.flatten()
    v = v / jnp.linalg.norm(v)
    # For each point, gather the k-nearest neighbor differences.
    A = coords[nn_idx] - coords[:, None, :]  # (Npts, k, 2)
    b = values[nn_idx] - values[:, None]      # (Npts, k)

    M = jnp.einsum('ijk,ijl->ikl', A, A)
    r = jnp.einsum('ijk,ij->ik', A, b)
    
    a = M[:, 0, 0]
    b_mat = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b_mat * c
    # For simplicity, assume det != 0 for all points.
    invM00 = d / det
    invM01 = -b_mat / det
    invM10 = -c / det
    invM11 = a / det
    grad0 = invM00 * r[:, 0] + invM01 * r[:, 1]
    grad1 = invM10 * r[:, 0] + invM11 * r[:, 1]
    grad = jnp.stack((grad0, grad1), axis=-1)
    deriv = jnp.dot(grad, v)
    return deriv  # (Npts,)

# Now, define a batched version using vmap.
# Precompute the NN indices for each batch.
def prepare_data_for_case(coords):
    nn_idx = compute_nn_idx(np.array(coords), k=5)  # using NumPy here
    return jnp.array(nn_idx)


def layer_norm(x, eps=1e-8, axis=-1):
    """
    Applies Layer Normalization over the specified axis.
    
    Parameters:
        x (jnp.array): The input tensor.
        gamma (jnp.array): Scale parameter. It should be broadcastable to the shape of x.
        beta (jnp.array): Shift parameter. It should be broadcastable to the shape of x.
        eps (float): A small constant to avoid division by zero.
        axis (int): The axis or axes over which normalization is computed.
        
    Returns:
        jnp.array: The layer-normalized output.
    """

    gamma = jnp.ones((x.shape[-1],))
    beta = jnp.zeros((x.shape[-1],))
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

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
        # inputsb = layer_norm(inputsb)
        # print("inputsb=\t",inputsb.shape)
        skip.append(inputsb)
    for i in range(1,L-1):
        skip[i] = jnp.add(skip[i],skip[i-1])
    # print("inputst0=\t",inputst.shape)

    for i in range(L-1):
        # print("inputst1=\t",inputst.shape)
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
        # inputst = layer_norm(inputst)
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

key = random.PRNGKey(1234)

key1, key2 =  random.split(key, num=2)

W_branch, b_branch = hyper_initial_WB(layers_f,key1)
a_branch, c_branch, a1_branch, F1_branch , c1_branch = hyper_initial_frequencies(layers_f)

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
    v, x = data

    # u_preds_der_x = batched_function(x, u_preds, normal_x, nn_idx_batch_jax)
    # u_preds_der_y = batched_function(x, u_preds, normal_y, nn_idx_batch_jax)

    # u_preds_der = jnp.concatenate((u_preds_der_x,u_preds_der_y),axis=-1)

    # u_preds_der = (u_preds_der - dermin)/(dermax - dermin)

    loss = jnp.mean((u_preds.flatten() - u.flatten())**2)
    # loss1 = jnp.mean((u_preds_der.flatten()-u_der.flatten())**2)

    mse = loss #+ 1e-1*loss1

    # print("loss=\t",loss)
    # print("loss1=\t",loss1)
    # print("loss2=\t",loss2)

    return mse

def loss_for_test(params, data, u):
    u_preds = predict(params, data)
    v, x = data

    # u_preds_der_x = batched_function_test(x, u_preds, normal_x, nn_idx_batch)
    # u_preds_der_y = batched_function_test(x, u_preds, normal_y, nn_idx_batch)

    # u_preds_der = jnp.concatenate((u_preds_der_x,u_preds_der_y),axis=-1)

    # u_preds_der = (u_preds_der - dermin)/(dermax - dermin)

    loss = jnp.mean((u_preds.flatten() - u.flatten())**2)
    # loss1 = jnp.mean((u_preds_der.flatten()-u_der.flatten())**2)

    mse = loss #+ 1e-1*loss1

    # der1 = jnp.abs((u_der-u_preds_der))
    # denom = jnp.abs(u_der)

    # l2_der = jnp.mean(jnp.linalg.norm(der1.flatten(), 2)/jnp.linalg.norm(denom.flatten() , 2))
    l2 = jnp.mean(jnp.linalg.norm(u.flatten() - u_preds.flatten(), 2)/jnp.linalg.norm(u.flatten() , 2))
    # print("loss=\t",loss)
    # print("loss1=\t",loss1)
    # print("loss2=\t",loss2)

    return mse,l2

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
u_tr , x_tr = jnp.array(u_train),jnp.array(x_train)
u_te , x_te = jnp.array(u_test),jnp.array(x_test)

# nn_idx_batch = np.stack([compute_nn_idx(x_tr[b], k=25) for b in range(x_tr.shape[0])], axis=0)
# nn_idx_batch_jax = jnp.array(nn_idx_batch)

# nn_idx_batch = np.stack([compute_nn_idx(x_te[b], k=25) for b in range(x_te.shape[0])], axis=0)
# nn_idx_batch_jax_te = jnp.array(nn_idx_batch)

# batched_function = jax.vmap(directional_derivative_single_case_jax, in_axes=(0, 0, None, 0))
# batched_function_test = jax.vmap(directional_derivative_single_case_jax, in_axes=(0, 0, None, 0))
# # jitted_function = jit(batched_function)
# u_tr_der_y = batched_function(x_tr, u_tr, normal_y, nn_idx_batch_jax)
# u_te_der_y = batched_function_test(x_te, u_te, normal_y, nn_idx_batch_jax_te)

# u_tr_der_x = batched_function(x_tr, u_tr, normal_x, nn_idx_batch_jax)
# u_te_der_x = batched_function_test(x_te, u_te, normal_x, nn_idx_batch_jax_te)

# u_tr_der = jnp.concatenate((u_tr_der_x,u_tr_der_y),axis=-1)
# u_te_der = jnp.concatenate((u_te_der_x,u_te_der_y),axis=-1)

# print("u_tr_der=\t",u_tr_der.shape)
# print("u_te_der=\t",u_te_der.shape)

# dermin = jnp.min(u_tr_der, axis=(0,1), keepdims=True)
# dermax = jnp.max(u_tr_der, axis=(0,1), keepdims=True)

# # u_tr_der = (u_tr_der - dermin)/(dermax - dermin)
# # u_te_der = (u_te_der - dermin)/(dermax - dermin)

# print("dermin=\t",dermin.shape)
# print("dermax=\t",dermax.shape)

# print("dermin1=\t",dermin)
# print("dermax1=\t",dermax)

# der_lim = [dermin,dermax]
# print("u_tr_der=\t",u_tr_der.shape)
# print("u_te_der=\t",u_te_der.shape)

epo = []
start_time = time.time()
n = 0

# loss_train= loss(params, [v_tr, x_tr],u_tr,u_tr_der)

# # u_train_pred = predict(params, [v_tr,x_tr])
start_time_tot = time.time()
for epoch in range(num_epochs_adam):
    epoch_time = time.time()
    # v_tr,x_tr,u_tr = shuffle_in_points(v_train,x_train,u_train)
    params, opt_state, loss_val = update(params, [v_tr,x_tr], u_tr, opt_state)
    losss = loss_val
    # print(epoch,losss)
    if epoch % 10 ==0:
        epoch_time = time.time() - start_time
        loss_test,l2_test= loss_for_test(params, [v_te, x_te],u_te)
        losss,l2_train= loss_for_test(params, [v_tr, x_tr],u_tr)
        
        train_loss.append(losss)
        test_loss.append(loss_test)
        epo.append(epoch)
        print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} | Test MSE: {:0.3e} | Train L2: {:0.5f} | Test L2: {:0.5f}".format(epoch, epoch_time,losss,loss_test,l2_train, l2_test))

    if epoch % 20 ==0:
        save_model(params,n)
        with open(filename_loss, 'wb') as file:
            pickle.dump((epo,train_loss,test_loss), file)
        n += 1
            
        # train_loss.append(l1)
        
    start_time = time.time()
    
time_tot = time.time() - start_time_tot
print("total time=\t",time_tot)

save_model(params,n)


with open(filename_loss, 'wb') as file:
    pickle.dump((epo,train_loss,test_loss), file)

