# %%
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Plotting
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# %%
#lecture code for rotation matrices 
def RotX(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[1, 1] = ct
    R[1, 2] = -st
    R[2, 1] = st
    R[2, 2] = ct
    return R

def RotY(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 2] = st
    R[2, 0] = -st
    R[2, 2] = ct
    return R

def RotZ(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 1] = -st
    R[1, 0] = st
    R[1, 1] = ct
    return R

def rot_inv(R):
    return R.T

# %%
def transform_point(pb, R, p = np.zeros((3, 1))):
    return p + R @ pb

def transform_vector(vb, R):
    return R @ vb

def homogeneous(R, p = np.zeros((3, 1))):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = p

    return T

# %%
#plot frame based on transformation matrices
def plot_frame(ax, T,length=1.0):
    origin = T[:3, 3:]
    R = T[:3, :3]
    xaxis = R @ np.array([[1., 0., 0.]]).T
    yaxis = R @ np.array([[0., 1., 0.]]).T
    zaxis = R @ np.array([[0., 0., 1.]]).T

    ax.quiver(*origin, *xaxis, color='r', length=length)
    ax.quiver(*origin, *yaxis, color='g', length=length)
    ax.quiver(*origin, *zaxis, color='b', length=length)
