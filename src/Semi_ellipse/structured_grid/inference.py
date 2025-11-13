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
from matplotlib.patches import Polygon


# def choose_model(thruth,inputs,mask,path):
#     l2_tot = []
#     files = []
#     for file in os.listdir(path):
#         if "model" in file:
#             filename = path+file
#             params = load_model(filename)
#             u_pred = predict(params, inputs)
#             u_pred = mask*u_pred
#             thruth = mask*thruth
#             u_pred1 = u_pred.flatten().copy()
#             thruth1 = thruth.flatten().copy()
#             l2 = np.mean(np.linalg.norm(thruth1 - u_pred1, 2)/np.linalg.norm(thruth1 , 2))
#             print(file)
#             print("l2 norm for model\t",file,"\t=\t",l2)
#             l2_tot.append(l2)
#             files.append(file)
#     ind = np.argmin(l2_tot)
#     model = files[ind]
#     print(model,"\t",l2_tot[ind])
#     return model

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Defining an optimizer in Jax
num_epochs_adam = 50000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 28
BS_t = 8
variables = [0,1,2,3] #rho
npts = 10000
# lr = 1e-3
scaling = '01'

# filename = "./models/model.4"
plot_dir = "./plots/"

path="./models/"

model = "model.4"

os.system("mkdir plots")

data = np.load("data.npz")

train = data["train"]
mask_train = data["mask_train"]
test = data["test"]
mask_test = data["mask_test"]
coord = data["coord"]
v_train = data["v_train"]
v_test = data["v_test"]

print("train=\t",train.shape)
print("mask_train=\t",train.shape)
print("test=\t",test.shape)
print("train=\t",train.shape)
print("coord=\t",coord.shape)

Xmin = np.min(v_train)
Xmax = np.max(v_train)

dmin = np.min(train,axis=(0,1,2))
dmax = np.max(train,axis=(0,1,2))


oness = np.ones((1,train.shape[1],train.shape[2]))
if scaling=='01':
    train = (train - dmin)/(dmax - dmin)
    test = (test- dmin)/(dmax - dmin)
    v_train = (v_train - Xmin)/(Xmax - Xmin) 
    v_test = (v_test- Xmin)/(Xmax - Xmin) 
else:
    train = 2*(train - dmin)/(dmax - dmin) - oness
    test = 2*(test- dmin)/(dmax - dmin) - oness
    v_train = 2.*(v_train - Xmin)/(Xmax - Xmin) - 1.0
    v_test = 2.*(v_test- Xmin)/(Xmax - Xmin) - 1.0

x_train = jnp.array(coord.reshape(-1, 2))
x_test  = jnp.array(coord.reshape(-1, 2))
u_train = jnp.array(train.reshape(len(train),-1, len(variables)))
u_test = jnp.array(test.reshape(len(test),-1, len(variables)))
v_train = jnp.array(v_train)
v_test = jnp.array(v_test)
mask_train = jnp.array(mask_train.reshape(len(train),-1, 1))
mask_test = jnp.array(mask_test.reshape(len(test),-1, 1))

print("x_train=\t",x_train.shape)
print("u_train=\t",u_train.shape)
print("v_train=\t",v_train.shape)
print("x_test\t",x_test.shape)
print("u_test=\t",u_test.shape)
print("v_test=\t",v_test.shape)

print("mask_train=\t",mask_train.shape)
print("mask_test=\t",mask_test.shape)


def fnn_fuse_oneway(Xt, Xb, pt,pb):
    Wt, bt, at, ct, a1t, F1t, c1t = pt
    Wb, bb, ab, cb, a1b, F1b, c1b = pb

    inputst = Xt
    inputsb = Xb
    L = len(Wt)
    skip = []
    for i in range(L-1):
        inputsb =  jnp.tanh(jnp.add(10*ab[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),cb[i])) \
            + 10*a1b[i]*jnp.sin(jnp.add(10*F1b[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),c1b[i]))
        skip.append(inputsb)
        
    for i in range(1,L-1):
        skip[i] = jnp.add(skip[i-1],skip[i])

    for i in range(L-1):
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
        
        # print("inputst=\t",inputst.shape)
        # print("skip=\t",skip[i][:,None,:].shape)
        # print("inputsb=\t",inputsb.shape)
        inputst = jnp.multiply(inputst,skip[i][:,None,:])
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

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params
    v, x = data


    u_out_trunk,u_out_branch = fnn_fuse_oneway(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch

    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)

    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)  
    return u_pred

def load_model(filename):
    with open(filename, 'rb') as file:
        params = pickle.load(file)
    return params



x_train = np.tile(x_train, (BS, 1, 1))
x_test = np.tile(x_test, (BS_t, 1, 1))
print("x_train=\t",x_train.shape)
print("x_test=\t",x_test.shape)

inputs = [v_test, x_test]


filename = path+model
params = load_model(filename)
u_pred = predict(params, [v_train, x_train])
u_pred_test = predict(params, [v_test, x_test])

dmin = dmin.reshape(1,1,len(variables),order='C')
dmax = dmax.reshape(1,1,len(variables),order='C')
if scaling=='01':
    u_pred = u_pred*(dmax - dmin)+dmin
    u_pred_test = u_pred_test*(dmax - dmin)+dmin
    v_train = v_train*(Xmax - Xmin)+Xmin
    v_test = v_test*(Xmax - Xmin)+Xmin
    u_train = u_train*(dmax - dmin)+dmin
    u_test= u_test*(dmax - dmin)+dmin
else:
    u_pred = 0.5*(u_pred+oness)*(dmax - dmin)+dmin
    u_pred_test = 0.5*(u_pred_test+oness)*(dmax - dmin)+dmin
    v_train = 0.5*(v_train+1)*(Xmax - Xmin)+Xmin
    v_test = 0.5*(v_test+1)*(Xmax - Xmin)+Xmin
    u_train = 0.5*(u_train+oness)*(dmax - dmin)+dmin
    u_test = 0.5*(u_test+oness)*(dmax - dmin)+dmin
pred   = u_pred_test.reshape(8,256,256,4,order='C')
target = u_test.reshape(8,256,256,4,order='C')
grid_x = x_train[0,:,0].reshape(256,256,order='C')
grid_y = x_train[0,:,1].reshape(256,256,order='C')
np.savez("Fusion_DeepONet_preds.npz", pred=pred, target=target, gridx=grid_x, gridy=grid_y)
diffs = np.abs(u_pred_test-u_test)*mask_test
denom = np.abs(u_test)*mask_test
l2 = []
for i in range(len(variables)):
    dd = diffs[:,:,i].flatten()
    de = denom[:,:,i].flatten()
    l2_test = np.linalg.norm(dd,2)/np.linalg.norm(de,2)
    l2.append(float(l2_test))
# 
print("l2=\t",l2)
# # Plot loss
file = "loss"
filename = os.path.join(path, file)  # Use os.path.join for platform-independent path construction
losses = load_model(filename)

epoch = losses[0]
loss_train = losses[1]
loss_test = losses[2]

plt.figure(figsize=(10, 6))
plt.plot(epoch, loss_train, label='Train')
plt.plot(epoch, loss_test, label='Test')

# Add labels with custom font sizes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
# plt.title('Loss', fontsize=14)

# Customize tick label font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.yscale('log')
plt.legend(loc='best', bbox_to_anchor=(0.95, 1), fontsize=16, ncol=1)  # Adjusted font size for legend

plt.tight_layout()
filename = os.path.join(plot_dir, "loss.png")
plt.savefig(filename, dpi=300)
plt.close()

def create_semi_ellipse_vertices(center, axes_lengths, orientation=0, num_points=100):
    """
    Create vertices for a semi-ellipse with given center, axes, and orientation.
    
    Args:
        center: Tuple (cx, cy) - the center of the ellipse.
        axes_lengths: Tuple (a, b) - lengths of the semi-major (a) and semi-minor (b) axes.
        orientation: Angle of rotation in radians.
        num_points: Number of points to approximate the semi-ellipse.
    
    Returns:
        Vertices of the semi-ellipse in the form of a (num_points, 2) array.
    """
    t = np.linspace(3*np.pi/2, 5*np.pi/2, num_points)  # Parametric angle for top half of ellipse
    a, b = axes_lengths
    cx, cy = center
    
    # Parametric equations for an ellipse
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotation matrix
    cos_angle = np.cos(orientation)
    sin_angle = np.sin(orientation)
    
    # Rotate and translate points
    x_rot = cos_angle * x - sin_angle * y + cx
    y_rot = sin_angle * x + cos_angle * y + cy
    
    vertices = np.column_stack((x_rot, y_rot))
    return vertices

def plot_sample(xx, yy, u_pred, u_truth, error, title_prefix, file, vmin, vmax,vertices,labels):
    """Plot with triangulation-based visualizations."""
    num_levels = 30
    fig, ax = plt.subplots(len(variables), 3, figsize=(15, 10))
    for j in range(len(variables)):
        levels = np.linspace(vmin[j], vmax[j], num_levels,endpoint=True)
        # First plot: Prediction
        tpc1 = ax[j,0].contourf(xx,yy, u_pred[:,:,j], vmin=vmin[j], vmax=vmax[j],levels=levels,cmap='inferno')
        ax[j,0].set_title(f'{labels[j]} {title_prefix} Output')
        cbar1 = fig.colorbar(tpc1, ax=ax[j,0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Second plot: Ground Truth
        tpc2 = ax[j,1].contourf(xx,yy, u_truth[:,:,j], vmin=vmin[j], vmax=vmax[j],levels=levels,cmap='inferno')
        ax[j,1].set_title(f'{labels[j]} {title_prefix} Ground Truth')
        cbar2 = fig.colorbar(tpc2, ax=ax[j,1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Third plot: Error
        tpc3 = ax[j,2].contourf(xx,yy, error[:,:,j],cmap='inferno')
        ax[j,2].set_title(f'{labels[j]} {title_prefix} Error')
        cbar3 = fig.colorbar(tpc3, ax=ax[j,2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.set_label('Error Color Scale')
        polygon = Polygon(vertices, closed=True, edgecolor='white', facecolor='white', lw=0.0)
        ax[j,0].add_patch(polygon)
        polygon = Polygon(vertices, closed=True, edgecolor='white', facecolor='white', lw=0.0)
        ax[j,1].add_patch(polygon)
        polygon = Polygon(vertices, closed=True, edgecolor='white', facecolor='white', lw=0.0)
        ax[j,2].add_patch(polygon)

    plt.tight_layout()
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"Saved plot: {file}")
center = (0.0, 0.0)  # Example center at (0.5, 0.5)
orientation = 0.0  # Example orientation (45 degrees)
labels = ["rho", "u", "v", "p"]

# temp = np.concatenate((u_train,u_test),axis=0)
# vmin = [0.0, -10.0, -2.5, 0.0]
# vmax = [8.5, 0.0, 2.5, 130.0]
vmin, vmax = u_train.min(axis=(0,1)), u_train.max(axis=(0,1))
print("u_train=\t",u_train.shape)
# print("vmin=\t",vmin.shape)
# vmin, vmax = u_truth_i1.min(axis=(0,1)), u_truth_i1.max(axis=(0,1))

# Loop for training set
# for i, (u_pred_i, u_truth_i, x_i,v_i) in enumerate(zip(u_pred, u_train, x_train,v_train)):
#     xx, yy = x_i[:, 0].reshape(256,256,order='C'), x_i[:, 1].reshape(256,256,order='C')
#     vmin, vmax = u_pred_i.min(axis=(0)), u_pred_i.max(axis=(0))
#     axes_lengths = v_i  # Semi-major and semi-minor axes lengths
#     # Create the mask for the semi-ellipse
#     vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
#     # print(vertices)
#     u_pred_i1 = u_pred_i.reshape(256,256,len(variables),order='C')
#     u_truth_i1 = u_truth_i.reshape(256,256,len(variables),order='C')
#     error = jnp.abs(u_pred_i1 - u_truth_i1)
#     # print("vmin=\t",vmin)
#     file = os.path.join(plot_dir, f"train_{i}.png")
#     plot_sample(xx, yy, u_pred_i1, u_truth_i1, error, "Train", file, vmin, vmax,vertices,labels)

#Loop for testing set
for i, (u_pred_test_i, u_truth_test_i, x_i, v_i) in enumerate(zip(u_pred_test, u_test, x_test, v_test)):
    xx, yy = x_i[:, 0].reshape(256,256,order='C'), x_i[:, 1].reshape(256,256,order='C')
    vmin, vmax = u_pred_test_i.min(axis=(0)), u_pred_test_i.max(axis=(0))
    axes_lengths = v_i  # Semi-major and semi-minor axes lengths
    # Create the mask for the semi-ellipse
    vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
    # print(vertices)
    u_pred_test_i = u_pred_test_i.reshape(256,256,len(variables),order='C')
    u_truth_test_i = u_truth_test_i.reshape(256,256,len(variables),order='C')
    error = jnp.abs(u_pred_test_i - u_truth_test_i)
    file = os.path.join(plot_dir, f"test_{i}.png")
    plot_sample(xx, yy, u_pred_test_i, u_truth_test_i, error, "Test", file, vmin, vmax,vertices,labels)
        