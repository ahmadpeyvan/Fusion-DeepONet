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
import matplotlib.tri as tri
from jax.example_libraries import optimizers

from matplotlib.patches import Polygon

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Defining an optimizer in Jax
num_epochs_adam = 1000+1
num_epochs_tot = num_epochs_adam  + 1
BS = 28
BS_t = 8
plot_dir = "plots_3_64"
os.system("mkdir "+plot_dir)
plot_dir = "./"+plot_dir+"/"
variables = [0,1,2,3] #T,k
npts = 10000
# lr = 1e-3
scaling = '01'

path = "./models/"
filename = "model.4"

def choose_model(thruth,inputs,path):
    l2_tot = []
    files = []
    for file in os.listdir(path):
        if "model" in file:
            filename = path+file
            params = load_model(filename)
            u_pred = predict(params, inputs)
            u_pred1 = u_pred.flatten().copy()
            thruth1 = thruth.flatten().copy()
            l2 = np.mean(np.linalg.norm(thruth1 - u_pred1, 2)/np.linalg.norm(thruth1 , 2))
            print(file)
            print("l2 norm for model\t",file,"\t=\t",l2)
            l2_tot.append(l2)
            files.append(file)
    ind = np.argmin(l2_tot)
    model = files[ind]
    print(model,"\t",l2_tot[ind])
    return model

# os.system('rm -r models')
# os.system('mkdir models')

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


def load_model(filename):
    with open(filename, 'rb') as file:
        params = pickle.load(file)
    return params

def fnn_fuse_mixed_add(Xt, Xb, pt,pb):
    Wt, bt, at, ct, a1t, F1t, c1t = pt
    Wb, bb, ab, cb, a1b, F1b, c1b = pb

    inputst = Xt
    inputsb = Xb
    skip = []
    L = len(Wt)
    for i in range(L-1):
        inputsb =  jnp.tanh(jnp.add(10*ab[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),cb[i])) \
            + 10*a1b[i]*jnp.sin(jnp.add(10*F1b[i]*jnp.add(jnp.dot(inputsb, Wb[i]), bb[i]),c1b[i]))
        skip.append(inputsb)
    
    for i in range(1,L-1):
        skip[i] = jnp.add(skip[i],skip[i-1])
    for i in range(L-1):
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
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
    u_out_trunk,u_out_branch = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)

    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)
    return u_pred

def predict_o(params, data, scaling, u_truth):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params
    v, x = data
    u_out_trunk,u_out_branch = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)

    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk)  
    if scaling=='01':
        u_pred  = u_pred*(dmax - dmin)+dmin
        u_truth = u_truth*(dmax - dmin)+dmin
        v = v*(Xmax - Xmin)+ Xmin
    else:
        u_pred = 0.5*(u_pred+oness)*(dmax - dmin)+dmin
        u_truth = 0.5*(u_truth+oness)*(dmax - dmin)+dmin
        v = 0.5*(v+oness)*(Xmax - Xmin)+Xmin
    return u_pred, u_truth, v 

def predict1(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk,c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch  = params
    v, x = data
    u_out_trunk,u_out_branch = fnn_fuse(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    u_out_branch = jnp.reshape(u_out_branch,(1,-1,len(variables)))
    u_pred = jnp.einsum('ijk,mj->imk',u_out_branch, u_out_trunk) 
    return u_pred

v_tr = jnp.array(v_train)
v_te = jnp.array(v_test)
u_tr , x_tr = jnp.array(u_tr),jnp.array(x_tr)
u_te , x_te = jnp.array(u_te),jnp.array(x_te)

inputs = [v_te, x_te]

thruth = u_te
# file = choose_model(thruth,inputs,path)

filename = path+filename
params = load_model(filename)
u_pred, u_truth, vtr  = predict_o(params, [v_tr,x_tr],scaling,u_tr)
u_pred_test, u_truth_test, vte  = predict_o(params, [v_te,x_te],scaling,u_te)

pred   = u_pred_test
target = u_truth_test
grid = x_te
np.savez("Fusion_DeepONet_preds_IG.npz", pred=pred, target=target, grid=grid)

# diffs = np.abs(u_pred_test-u_truth_test)
# denom = np.abs(u_truth_test)
# l2 = []
# for i in range(len(variables)):
#     dd = diffs[:,:,i].flatten()
#     de = denom[:,:,i].flatten()
#     l2_test = np.linalg.norm(dd,2)/np.linalg.norm(de,2)
#     l2.append(float(l2_test))

# print("l2=\t",l2)

# Plot loss
file = "loss"
filename = os.path.join(path, file)  # Use os.path.join for platform-independent path construction
losses = load_model(filename)

# epoch = np.array(range(0, 50001, 10))
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
    t = np.linspace(0, 2*np.pi, num_points)  # Parametric angle for top half of ellipse
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

def plot_sample_scatter(xx, yy, u_pred, u_truth, error, title_prefix, file, vmin, vmax, vertices, labels):
    """Plot with scatter-based visualizations."""
    num_vars = len(labels)
    fig, ax = plt.subplots(num_vars, 3, figsize=(15, 12))
    
    for j in range(num_vars):
        # Prediction scatter
        sc1 = ax[j,0].scatter(
            xx, yy,
            c=u_pred[:, j],
            cmap='inferno',
            vmin=vmin[j],
            vmax=vmax[j],
            s=1,          # you can adjust marker size
            marker='o'
        )
        ax[j,0].set_title(f'{labels[j]} {title_prefix} Output')
        ax[j,0].set_xlim(0,5)
        cbar1 = fig.colorbar(sc1, ax=ax[j,0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Ground truth scatter
        sc2 = ax[j,1].scatter(
            xx, yy,
            c=u_truth[:, j],
            cmap='inferno',
            vmin=vmin[j],
            vmax=vmax[j],
            s=1,
            marker='o'
        )
        ax[j,1].set_title(f'{labels[j]} {title_prefix} Ground Truth')
        ax[j,1].set_xlim(0,5)
        cbar2 = fig.colorbar(sc2, ax=ax[j,1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Error scatter
        sc3 = ax[j,2].scatter(
            xx, yy,
            c=error[:, j],
            cmap='inferno',
            s=1,
            marker='o'
        )
        ax[j,2].set_title(f'{labels[j]} {title_prefix} Error')
        ax[j,2].set_xlim(0,5)
        cbar3 = fig.colorbar(sc3, ax=ax[j,2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.set_label('Error Color Scale')

        # Mask out the interior by overlaying a white polygon
        # for col in range(3):
        #     poly = Polygon(vertices, closed=True, edgecolor='white', facecolor='white', lw=0)
        #     ax[j, col].add_patch(poly)

    plt.tight_layout()
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"Saved plot: {file}")

def plot_sample_triangulation(xx, yy, u_pred, u_truth, error, title_prefix, file, vmin, vmax,vertices,labels):
    """Plot with triangulation-based visualizations."""
    # Calculate the ranges of x and 

    # Create a triangulation
    triang = tri.Triangulation(xx.ravel() , yy.ravel() )
    # Mask the data: set values inside the semi-ellipse to Na
    # print("u_predin=\t",len(variables))
    # Create a figure with dynamic size
    fig, ax = plt.subplots(len(variables), 3, figsize=(15, 10))
    for j in range(len(variables)):
        # First plot: Prediction
        tpc1 = ax[j,0].tripcolor(triang, u_pred[:,j].ravel() , shading='flat', vmin=vmin[j], vmax=vmax[j],cmap='inferno')
        ax[j,0].set_title(f'{labels[j]} {title_prefix} Output')
        cbar1 = fig.colorbar(tpc1, ax=ax[j,0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Second plot: Ground Truth
        tpc2 = ax[j,1].tripcolor(triang, u_truth[:,j].ravel() , shading='flat', vmin=vmin[j], vmax=vmax[j],cmap='inferno')
        ax[j,1].set_title(f'{labels[j]} {title_prefix} Ground Truth')
        cbar2 = fig.colorbar(tpc2, ax=ax[j,1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label(f'{labels[j]} {title_prefix} Color Scale')

        # Third plot: Error
        tpc3 = ax[j,2].tripcolor(triang, error[:,j].ravel() , shading='flat',cmap='inferno')
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
# Loop for training set
# for i, (u_pred_i, u_truth_i, x_i,v_i) in enumerate(zip(u_pred, u_truth, x_tr,vtr)):
#     xx, yy = x_i[:, 0], x_i[:, 1]
    
#     axes_lengths = v_i  # Semi-major and semi-minor axes lengths
#     # Create the mask for the semi-ellipse
#     vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
#     # print(vertices)
#     error = jnp.abs(u_pred_i - u_truth_i)
#     vmin, vmax = u_truth_i.min(axis=0), u_truth_i.max(axis=0)
#     # print("vmin=\t",vmin)
#     file = os.path.join(plot_dir, f"train_{i}.png")
#     plot_sample_scatter(xx, yy, u_pred_i, u_truth_i, error, "Train", file, vmin, vmax,vertices,labels)

# Loop for testing set
for i, (u_pred_test_i, u_truth_test_i, x_i, v_i) in enumerate(zip(u_pred_test, u_truth_test, x_te, vte)):
    xx, yy = x_i[:, 0], x_i[:, 1]
    axes_lengths = v_i  # Semi-major and semi-minor axes lengths
    # Create the mask for the semi-ellipse
    vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
    error = jnp.abs(u_pred_test_i - u_truth_test_i)
    vmin, vmax = u_truth_test_i.min(axis=0), u_truth_test_i.max(axis=0)
    file = os.path.join(plot_dir, f"test_{i}.png")
    # plot_sample(xx, yy, u_pred_i, u_truth_i, error, "Test", file, vmin, vmax)
    plot_sample_scatter(xx, yy, u_pred_test_i, u_truth_test_i, error, f"Test_{i}", file, vmin, vmax,vertices,labels)
        