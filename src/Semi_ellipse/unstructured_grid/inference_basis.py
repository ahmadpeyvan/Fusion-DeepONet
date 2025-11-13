import jax
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
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
plot_dir = "plots_3_64_basis"
os.system("mkdir "+plot_dir)
plot_dir = "./"+plot_dir+"/"
variables = [0,1,2,3] #T,k
npts = 10000
# lr = 1e-3
scaling = '01'
# filename = "./models/model.32" # mutiply
path = "./models/"
filename = "model.4"


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

def plot_basis(xx, yy, u_pred, title_prefix, file, vmin, vmax):
    """Plot with triangulation-based visualizations."""
    # Calculate the ranges of x and 

    # Create a triangulation
    triang = tri.Triangulation(xx.ravel() , yy.ravel() )
    # Mask the data: set values inside the semi-ellipse to Na
    # print("u_predin=\t",len(variables))
    # Create a figure with dynamic size
    fig, ax = plt.subplots(1, figsize=(15, 10))

    # First plot: Prediction
    tpc1 = ax.tripcolor(triang, u_pred[:].ravel() , shading='flat', vmin=vmin, vmax=vmax,cmap='inferno')
    ax.set_title(f'{title_prefix} Output', fontsize=30)
    ax.tick_params(axis='both', labelsize=24)
    cbar1 = fig.colorbar(tpc1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label(f' {title_prefix} Color Scale', fontsize=30)

    cbar1.ax.tick_params(labelsize=30)           # tick labels font size
    formatter = ticker.FormatStrFormatter('%.4f')  # change 3 to 2 for 2 decimals
    cbar1.ax.yaxis.set_major_formatter(formatter)

    polygon = Polygon(vertices, closed=True, edgecolor='white', facecolor='white', lw=0.0)
    ax.add_patch(polygon)

    plt.tight_layout()
    plt.savefig(file, dpi=300)
    plt.close()
    print(f"Saved plot: {file}")


# def plot_basis(xx, yy, u_pred, title_prefix, file, vmin, vmax):
#     """Plot with triangulation-based visualizations."""
#     num_levels = 30
#     fig, ax = plt.subplots(1, figsize=(15, 10))
#     # for j in range(len(variables)):
#     levels = np.linspace(vmin, vmax, num_levels,endpoint=True)
#     # First plot: Prediction
#     tpc1 = ax.contourf(xx,yy, u_pred[:,:], vmin=vmin, vmax=vmax,levels=levels,cmap='inferno')
#     ax.set_title(f'{title_prefix} Output')
#     cbar1 = fig.colorbar(tpc1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
#     cbar1.set_label(f'Color Scale')
#     plt.tight_layout()
#     plt.savefig(file, dpi=300)
#     plt.close()
#     print(f"Saved plot: {file}")

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

# u_train =  u_train[:,0:1]
# u_test =  u_test[:,0:1]

# np_train = np_train.flatten()[0:,None]
# np_test = np_test.flatten()[0:,None]
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

    trunk_out = []
    for i in range(L-1):
        # print("inputst1=\t",inputst.shape)
        inputst =  jnp.tanh(jnp.add(10*at[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),ct[i])) \
            + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*jnp.add(jnp.dot(inputst, Wt[i]), bt[i]),c1t[i]))
        trunk_out.append(inputst)
        inputst = jnp.multiply(inputst,skip[i][:,None,:])
        
    Yt = jnp.dot(inputst, Wt[-1]) + bt[-1]     
    Yb = jnp.dot(inputsb, Wb[-1]) + bb[-1]
    return Yt, Yb,trunk_out
    
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
    u_out_trunk,u_out_branch,trunk_out_l = fnn_fuse_mixed_add(x,v,[W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]
    ,[W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch, c1_branch]) # predict on branch
    u_out_branch = jnp.reshape(u_out_branch,(u_out_branch.shape[0],-1,len(variables)))
    u_out_branch = jnp.expand_dims(u_out_branch,axis=1)
    u_out_trunk = jnp.expand_dims(u_out_trunk,axis=-1)

    u_pred = jnp.einsum('ijkl,inkm->inl',u_out_branch, u_out_trunk) 
    return u_pred,trunk_out_l

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

filename = path+filename
params = load_model(filename)
u_pred, trunk_out_l = predict(params, [v_tr,x_tr])
u_pred_test, trunk_out_l_test  = predict(params, [v_te,x_te])

v_train = v_train*(Xmax - Xmin)+Xmin#(v_train - Xmin)/(Xmax - Xmin) 
# v_test = (v_test- Xmin)/(Xmax - Xmin)

center = (0.0, 0.0)  # Example center at (0.5, 0.5)
orientation = 0.0  # Example orientation (45 degrees)
eigens = []
for j in range(28):
    temp=[]
    axes_lengths = v_train[j,:]  # Semi-major and semi-minor axes lengths
    vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
    for i in range(len(trunk_out_l)):
        mat = trunk_out_l[i][j,:,:]
        U, S, Vt = np.linalg.svd(mat,full_matrices=False)
        x = np.arange(64)
        maxx,minn=np.max(S),np.min(S)
        Scale = (S-minn)/(maxx-minn)
        temp.append(Scale)
        for k in range(G_dim):
            xx, yy = x_tr[j, :,0], x_tr[j, :,1]
            vmin, vmax = U[:,k].min(axis=(0)), U[:,k].max(axis=(0))
            # axes_lengths = v_i  # Semi-major and semi-minor axes lengths
            # Create the mask for the semi-ellipse
            # vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
            # print(vertices)
            basis = U[:,k]
            if i==2 and j==23:
                file = os.path.join(plot_dir, f"basis_{i}_{j}_{k}.png")
                plot_basis(xx, yy, basis, "Basis", file, vmin, vmax)

#     eigens.append(temp)
# rang = [[1e-9,10.0],[1e-5,10.0],[1e-4,10.0]]
# for i in range(3):
#     plt.figure(figsize=(10, 6))
#     for j in range(28):
#         plt.plot(x, eigens[j][i], label=f'case {j+1}')
#     plt.xlabel('Modes',fontsize=16)
#     plt.ylabel('Spectrum',fontsize=16)
#     plt.yscale('log')
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.ylim(rang[i])
#     plt.title(f'Fusion DeepONet Spectrum for layer {i+1}')
#     plt.legend(loc='best', bbox_to_anchor=(1.2, 1), fontsize='small', ncol=1)
#     plt.tight_layout()
#     filename = os.path.join(plot_dir, f"spectrum{j}_{i}_fd_us.png")
#     plt.savefig(filename,dpi=300)
#     plt.close()


# # diffs = np.abs(u_pred_test-u_truth_test)
# # denom = np.abs(u_truth_test)
# # l2 = []
# # for i in range(len(variables)):
# #     dd = diffs[:,:,i].flatten()
# #     de = denom[:,:,i].flatten()
# #     l2_test = np.linalg.norm(dd,2)/np.linalg.norm(de,2)
# #     l2.append(float(l2_test))

# # print("l2=\t",l2)

# # l2_cases = []

# # for j in range(diffs.shape[0]):
# #     temp = []
# #     for i in range(len(variables)):
# #         dd = diffs[j,:,i]
# #         de = denom[j,:,i]
# #         l2_test = float(np.linalg.norm(dd,2)/np.linalg.norm(de,2))
# #         temp.append(l2_test)
# #     l2_cases.append(temp)
# # print("l2=\t",l2)

# # l2_cases = np.array(l2_cases)
# # np.savez("l2_test_FusionDeepONet_IG.npz")
# # print("l2_cases=\t",l2_cases.shape)
# # print("l2_cases=\t",l2_cases)
# # def plot_sample(xx, yy, u_pred, u_truth, error, title_prefix, file, vmin, vmax):
# #     """Utility to create and save scatter plots for prediction, ground truth, and error."""

# #     # Create a figure with dynamic size
# #     fig, ax = plt.subplots(1, 3, figsize=(15,10))

# #     # First scatter plot: Prediction
# #     sc1 = ax[0].scatter(xx, yy, s=1, c=u_pred, vmin=vmin, vmax=vmax)
# #     ax[0].set_title(f'{title_prefix} Output')
# #     cbar1 = fig.colorbar(sc1, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
# #     cbar1.set_label(f'{title_prefix} Color Scale')

# #     # Second scatter plot: Ground Truth
# #     sc2 = ax[1].scatter(xx, yy, s=1, c=u_truth, vmin=vmin, vmax=vmax)
# #     ax[1].set_title(f'{title_prefix} Ground Truth')
# #     cbar2 = fig.colorbar(sc2, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
# #     cbar2.set_label(f'{title_prefix} Color Scale')

# #     # Third scatter plot: Error
# #     sc3 = ax[2].scatter(xx, yy, s=1, c=error)
# #     ax[2].set_title(f'{title_prefix} Error')
# #     cbar3 = fig.colorbar(sc3, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
# #     cbar3.set_label('Error Color Scale')

# #     plt.tight_layout()
# #     plt.savefig(file, dpi=300)
# #     plt.close()
# #     print(f"Saved plot: {file}")



# # labels = ["rho", "u", "v", "p"]
# # # Loop for training set
# # for i, (u_pred_i, u_truth_i, x_i,v_i) in enumerate(zip(u_pred, u_truth, x_tr,vtr)):
# #     xx, yy = x_i[:, 0], x_i[:, 1]
    
# #     axes_lengths = v_i  # Semi-major and semi-minor axes lengths
# #     # Create the mask for the semi-ellipse
# #     vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
# #     # print(vertices)
# #     error = jnp.abs(u_pred_i - u_truth_i)
# #     vmin, vmax = u_truth_i.min(axis=0), u_truth_i.max(axis=0)
# #     # print("vmin=\t",vmin)
# #     file = os.path.join(plot_dir, f"train_{i}.png")
# #     plot_sample_triangulation(xx, yy, u_pred_i, u_truth_i, error, "Train", file, vmin, vmax,vertices,labels)

# # # Loop for testing set
# # for i, (u_pred_test_i, u_truth_test_i, x_i, v_i) in enumerate(zip(u_pred_test, u_truth_test, x_te, vte)):
# #     xx, yy = x_i[:, 0], x_i[:, 1]
# #     axes_lengths = v_i  # Semi-major and semi-minor axes lengths
# #     # Create the mask for the semi-ellipse
# #     vertices = create_semi_ellipse_vertices(center, axes_lengths, orientation)
# #     error = jnp.abs(u_pred_test_i - u_truth_test_i)
# #     vmin, vmax = u_truth_test_i.min(axis=0), u_truth_test_i.max(axis=0)
# #     file = os.path.join(plot_dir, f"test_{i}.png")
# #     plot_sample_triangulation(xx, yy, u_pred_test_i, u_truth_test_i, error, "Test", file, vmin, vmax,vertices,labels)
# # #         