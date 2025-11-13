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
import glob
import re

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from jax import jacrev, jacfwd


from jax.example_libraries import optimizers

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Defining an optimizer in Jax
num_epochs_adam = 50000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 42
BS_t = 18
variables = [4] #rho
index_test = [23, 30, 46,  2, 27, 17, 51, 16, 21, 49,  0,  4]
# index_test = [23, 30, 46,  2, 27, 17, 51, 16, 21, 49,  0,  4]
npts = 10000
# lr = 1e-3
scaling = '01'

# modelfile = "./models/model.4945" ##this happeed for 80-20
modelfile = "../Training/models_autodiff/model.5000"
# modelfile = "./models_autodiff/model.4687"

data_train = np.load("../Training/training_dataset.npz")
data_test = np.load("../Training/testing_dataset.npz")

nv = len(variables)

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

print("dmin=\t",dmin.shape)
print("dmax=\t",dmax.shape)

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

def load_model(filename):
    with open(filename, 'rb') as file:
        params = pickle.load(file)
    return params

initializer = jax.nn.initializers.glorot_normal()

# def hyper_initial_WB(layers,key):
#     L = len(layers)
#     W = []
#     b = []
#     for l in range(1, L):
#         in_dim = layers[l-1]
#         out_dim = layers[l]
#         std = np.sqrt(2.0/(in_dim+out_dim))
#         weight = initializer(key, (in_dim, out_dim), jnp.float32)*std
#         bias = initializer(key, (1, out_dim), jnp.float32)*std
#         W.append(weight)
#         b.append(bias)
#     return W, b

# def hyper_parameters_A(shape):
#     return jnp.full(shape, 0.1, dtype=jnp.float32)

# def hyper_parameters_amplitude(shape):
#     return jnp.full(shape, 0.0, dtype=jnp.float32)

# def hyper_parameters_freq1(shape):
#     return jnp.full(shape, 0.1, dtype=jnp.float32)

 

# def hyper_initial_frequencies(layers):

#     L = len(layers)

#     a = []
#     c = []

#     a1 = []
#     F1 = []
#     c1 = []
    
#     for l in range(1, L):

#         a.append(hyper_parameters_A([1]))
#         c.append(hyper_parameters_A([1]))

#         a1.append(hyper_parameters_amplitude([1]))
#         F1.append(hyper_parameters_freq1([1]))
#         c1.append(hyper_parameters_amplitude([1]))

#     return a, c, a1, F1, c1 

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
        # print("skip=\t",skip[i].,"\t",i)

    # if not hasattr(fnn_fuse_mixed_add, "has_run"):
    #     # Code here runs only on the very first call.
    #     print("This block runs only once.")
    #     fnn_fuse_mixed_add.has_run = True  # Set the flag.
    #     for i in range(len(skip)):
    #         np.savetxt("skips"+str(i)+".txt",skip[i])

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
layers_f = [u_dim] + [64]*5 + [len(variables)*G_dim]
# print("branch layers:\t",layers_f)

# Trunk dim
x_dim = 2

#Trunk Net
layers_x = [x_dim] + [64]*5 + [G_dim]
# print("Trunks layers:\t",layers_x)
# key = random.PRNGKey(1234)
# # key = random.PRNGKey(6534)
# key1, key2 =  random.split(key, num=2)

# W_branch, b_branch = hyper_initial_WB(layers_f,key1)
# a_branch, c_branch, a1_branch, F1_branch , c1_branch = hyper_initial_frequencies(layers_f)
# # key1 = random.PRNGKey(6534)
# W_trunk, b_trunk = hyper_initial_WB(layers_x,key2)
# a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk = hyper_initial_frequencies(layers_x)

# def predict(params, data,scaling, u_truth):
#     W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params
#     v, x = data

#     # print("v=\t",v.shape)
#     # print("x=\t",x.shape)
#     # print("u_truth=\t",u_truth.shape)

#     u_out_trunk,u_out_branch = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
#     ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
#     # print("u_out_trunk=\t",u_out_trunk.shape)
#     # print("u_out_branch1=\t",u_out_branch.shape)
#     u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
#     u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
#     u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)
#     # print("u_out_branch2=\t",u_out_branch.shape)
#     # print("u_out_trunk2=\t",u_out_trunk.shape)
#     u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)  

#     # print("u_pred=\t",u_pred.shape)

#     if scaling=='01':
#         u_truth = u_truth*(dmax - dmin)+dmin
#         u_pred  = u_pred*(dmax - dmin)+dmin
#         v = v*(Xmax - Xmin)+ Xmin
#     else:
#         u_truth = 0.5*(u_truth+oness)*(dmax - dmin)+dmin
#         u_pred = 0.5*(u_pred+oness)*(dmax - dmin)+dmin
#         v = 0.5*(v+oness)*(Xmax - Xmin)+Xmin

#     # print("u_pred=\t",u_pred.shape)
#     return u_pred,u_truth,v

def fnn_fuse(Xt, Xb, pt,pb):
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
        inputst = jnp.multiply(inputst,skip[i])
        # inputst = jnp.multiply(inputst,skip_time[i][None,:, None,:])
        
    Yt = jnp.dot(inputst, Wt[-1]) + bt[-1]     
    Yb = jnp.dot(inputsb, Wb[-1]) + bb[-1]

    return Yt, Yb

def predict(params, x, v):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params

    Yt,Yb = fnn_fuse(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch])

    Yb = Yb.reshape(-1,nv)
    pred = jnp.sum(jnp.multiply(Yt.squeeze()[:,None],Yb),axis=0)

    return pred

predict_pts_vect = vmap(predict,in_axes=(None, 0, None))
predict_sample_vect = vmap(predict_pts_vect,in_axes=(None, 0, 0))

# vectorizing the automatic differentiation function
jac_fn_rev = jacrev(predict, argnums=1)
jac_fn_rev_pts_vect = vmap(jac_fn_rev,in_axes=(None, 0, None))
jac_fn_rev_sample_vect = vmap(jac_fn_rev_pts_vect,in_axes=(None, 0, 0))

# Define a wrapper function that computes T from the model for a single spatial coordinate.
# def T_fn(x_input,v_input,dummy_u_truth):
#     x_input=jnp.expand_dims(x_input, axis=(0,1))
#     v_input=jnp.expand_dims(v_input, axis=(0))
#     dummy_u_truth=jnp.expand_dims(dummy_u_truth, axis=(0,1))
#     u_pred, _, _ = predict(params, [v_input, x_input], scaling, dummy_u_truth)
#     # Extract T from the predicted u. Here we assume T is stored in the first channel.
#     T_value = u_pred[0, 0, 4]
#     return T_value

# def u_fn(x_input,v_input,dummy_u_truth):
#     x_input=jnp.expand_dims(x_input, axis=(0,1))
#     v_input=jnp.expand_dims(v_input, axis=(0))
#     dummy_u_truth=jnp.expand_dims(dummy_u_truth, axis=(0,1))
#     u_pred, _, _ = predict(params, [v_input, x_input], scaling, dummy_u_truth)
#     # Extract T from the predicted u. Here we assume T is stored in the first channel.
#     T_value = u_pred[0, 0, 1]
#     return T_value

# def v_fn(x_input,v_input,dummy_u_truth):
#     x_input=jnp.expand_dims(x_input, axis=(0,1))
#     v_input=jnp.expand_dims(v_input, axis=(0))
#     dummy_u_truth=jnp.expand_dims(dummy_u_truth, axis=(0,1))
#     u_pred, _, _ = predict(params, [v_input, x_input], scaling, dummy_u_truth)
#     # Extract T from the predicted u. Here we assume T is stored in the first channel.
#     T_value = u_pred[0, 0, 2]
#     return T_value

# def grad_T_fn(x_input,v_input,dummy_u_truth):
#     return jax.grad(T_fn,argnums=0,has_aux=False)(x_input,v_input,dummy_u_truth)

# def grad_u_fn(x_input,v_input,dummy_u_truth):
#     return jax.grad(u_fn,argnums=0,has_aux=False)(x_input,v_input,dummy_u_truth)

# def grad_v_fn(x_input,v_input,dummy_u_truth):
#     return jax.grad(v_fn,argnums=0,has_aux=False)(x_input,v_input,dummy_u_truth)

# gradsT = jax.vmap(grad_T_fn)
# gradsu = jax.vmap(grad_u_fn)
# gradsv = jax.vmap(grad_v_fn)

v_tr = jnp.array(v_train)
v_te = jnp.array(v_test)
# u_tr , x_tr = jnp.array(u_train),jnp.array(x_train)
# u_te , x_te = jnp.array(u_test),jnp.array(x_test)
params = load_model(modelfile)
# u_pred, u_truth, v  = predict_o(params, [v_tr,x_tr],scaling,u_tr)
# u_pred_test, u_truth_test, v  = predict_o(params, [v_te,x_te],scaling,u_te)
pattern = re.compile(r"^solution_\d+\.vtu$")
i = 0
l2_norm = []
for ind in index_test:
    print("ind=\t",ind)
    # folder = f"../cases_interp_high/out_{ind}"
    folder = f"../cases/out_{ind}"
    vtu_files = glob.glob(os.path.join(folder, "*.vtu"))
    clean_files = [
    f for f in vtu_files
    if pattern.match(os.path.basename(f))
    ]
    print("vtu_files=\t",clean_files[0])
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(clean_files[0])
    reader.Update()
    
    # Get the unstructured grid output
    data = reader.GetOutput()
    
    # Extract coordinates
    points = data.GetPoints()
    
    point_data = data.GetPointData()
    rho = vtk_to_numpy(point_data.GetArray("rho"))
    v1  = vtk_to_numpy(point_data.GetArray("v1"))
    v2  = vtk_to_numpy(point_data.GetArray("v2"))
    p   = vtk_to_numpy(point_data.GetArray("p"))

    T = p* 1.4 * 850/(rho* 0.014103202 * 287.058)

    # num_points = points.GetNumberOfPoints()
    coords = vtk_to_numpy(points.GetData())[:,0:2]
    x_te = jnp.array(coords[None,:,:])
    u_te = jnp.array(u_test[i:i+1,:,:])

    u_pred  = predict_sample_vect(params, x_te, v_te[i:i+1,:])
    u_preds_der = jac_fn_rev_sample_vect(params, x_te, v_te[i:i+1,:])


    if scaling=='01':
        # u_truth = u_truth*(dmax - dmin)+dmin
        u_pred  = u_pred*(dmax - dmin)+dmin
        u_preds_der  = u_preds_der*(dmax[:,:,:,None] - dmin[:,:,:,None])+dmin[:,:,:,None]
        # v = v*(Xmax - Xmin)+ Xmin
    else:
        # u_truth = 0.5*(u_truth+oness)*(dmax - dmin)+dmin
        u_pred = 0.5*(u_pred+oness)*(dmax - dmin)+dmin
        # v = 0.5*(v+oness)*(Xmax - Xmin)+Xmin
    u_preds_der = u_preds_der.squeeze()
    print("u_pred=\t",u_pred.shape)
    print("u_preds_der=\t",u_preds_der.shape)

    # u1_pred = u_pred[0,:,0]
    # v_pred = u_pred[0,:,1]
    # p_pred = u_pred[0,:,3]
    T_pred = u_pred[0,:,0]
    u_truth = np.concatenate((v1[:,None],v2[:,None],T[:,None]),axis=-1)
    u_truth = u_truth[None,:,:]
    # rho_error=np.abs(u_truth[0,:,0]-rho_pred)
    # u_error=np.abs(u_truth[0,:,0]-u1_pred)
    # v_error=np.abs(u_truth[0,:,1]-v_pred)
    T_error=np.abs(u_truth[0,:,2]-T_pred)/np.abs(u_truth[0,:,2])

    l2 = np.linalg.norm(u_truth[0,:,2]-T_pred)/np.linalg.norm(u_truth[0,:,2])
    l2_norm.append(l2)

    print("T_error=\t",np.mean(T_error),"\t",np.max(T_error))
    
    x = x_te.squeeze()
    print("x=\t",x.shape)
    v=v_te[i:i+1,:]
    print("v=\t",v.shape)
    v = jnp.repeat(v, x.shape[0], axis=0)
    print("v=\t",v.shape)
    print("u_te=\t",u_te.shape)
    u_truth = u_te[0,0:1,:]
    print("u_truth=\t",u_truth.shape)
    u_truth=jnp.repeat(u_truth, x.shape[0], axis=0)

    print("bbu_truth=\t",u_truth.shape)
    print("bbv=\t",v.shape)
    print("bbx=\t",x.shape)
    # DelT = gradsT(x,v,u_truth)
    vec = jnp.zeros((u_preds_der.shape[0],1))
    print("vec=\t",vec.shape)
    gradT=np.concatenate((u_preds_der[:,:],vec),axis=-1)
    print("gradT=\t",gradT.shape)

    # l2 = np.linalg.norm(gradT[:,0]-gradT_pred[:,0])/np.linalg.norm(gradT[:,0])
    # l2_norm_Tx.append(l2)
    # l2 = np.linalg.norm(gradT[:,1]-gradT_pred[:,1])/np.linalg.norm(gradT[:,1])
    # l2_norm_Ty.append(l2)


    vtk_vector = numpy_to_vtk(gradT)
    vtk_vector.SetName("gradT_pred")  # Name the array (e.g., "velocity")
    vtk_vector.SetNumberOfComponents(3)


    Tvtk = numpy_to_vtk(T_pred)
    Tvtk.SetName("T_pred")
    Ter = numpy_to_vtk(T_error)
    Ter.SetName("T_err")

    Tvtke = numpy_to_vtk(T)
    Tvtke.SetName("T")


    point_data = data.GetPointData()
    # point_data.AddArray(rhovtk)
    # point_data.AddArray(uvtk)
    # point_data.AddArray(vvtk)
    # point_data.AddArray(pvtk)
    point_data.AddArray(Tvtk)
    point_data.AddArray(Tvtke)

    # point_data.AddArray(rhoer)
    # point_data.AddArray(uer)
    # point_data.AddArray(ver)
    # point_data.AddArray(per)
    point_data.AddArray(Ter)

    point_data.AddArray(vtk_vector)
    # point_data.AddArray(vtk_vector_u)
    # point_data.AddArray(vtk_vector_v)


    writer = vtk.vtkXMLUnstructuredGridWriter()
    outfile = "case_"+str(ind)+".vtu"
    writer.SetFileName(outfile)
    writer.SetInputData(data)
    writer.Write()
    i +=1


l2_mean = np.mean(np.array(l2_norm))
print("l2_norm, var=\t",l2_mean,"\t",np.var(l2_norm))



