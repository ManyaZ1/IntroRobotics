import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
#  1) Utilities: Skew, Adjoint, SE(3) Exponential
##############################################################################
def skew_3(w):
    """3x3 skew-symmetric matrix from vector w."""
    return np.array([
        [    0., -w[2],  w[1]],
        [ w[2],     0., -w[0]],
        [-w[1],  w[0],    0.]
    ])

def adjoint(T):
    """
    Ad(T) for a 4x4 transform => 6x6
    T = [[R, p],
         [0,  1]]
    """
    R = T[0:3,0:3]
    p = T[0:3, 3]
    AdT = np.zeros((6,6))
    AdT[0:3,0:3] = R
    AdT[3:6,3:6] = R
    AdT[3:6,0:3] = skew_3(p) @ R
    return AdT

def exp_se3(xi, theta):
    """
    Exponential in SE(3). xi=[w,v]. If ||w||=0 => prismatic.
    Otherwise revolve about w.
    """
    w = xi[0:3]
    v = xi[3:6]
    normw = np.linalg.norm(w)
    if normw<1e-8:
        # prismatic
        T= np.eye(4)
        T[0:3,3] = v*theta
        return T

    # revolve
    w_unit= w/normw
    W= skew_3(w_unit)
    c= np.cos(theta)
    s= np.sin(theta)
    I3= np.eye(3)
    R= I3 + s*W + (1-c)*(W@W)
    w_cross_v= np.cross(w_unit,v)
    p= (I3 - R)@ w_cross_v + w_unit*(w_unit.dot(v))*theta
    T= np.eye(4)
    T[0:3,0:3]= R
    T[0:3,3]  = p
    return T

##############################################################################
#  2) Robot Data (4 joints) and "Home" Transforms M_list
##############################################################################
def make_transform_from_z_offset(z):
    T= np.eye(4)
    T[2,3]= z
    return T

# Suppose each joint i is at [0,0,z_i].
# We'll define M_{i+1,i} by the difference in z:
joints_data= [
    ('joint1',[0,0,0.0545]),
    ('joint2',[0,0,0.1775]),
    ('joint3',[0,0,0.4945]),
    ('joint4',[0,0,0.6965])
]
M_list=[]
for i in range(4):
    z_i= joints_data[i][1][2]
    z_im1= joints_data[i-1][1][2] if i>0 else 0.
    offset= z_i- z_im1
    M_list.append(make_transform_from_z_offset(offset))

# Screw axes
A1= np.array([0,0,1, 0,0,0])
A2= np.array([0,1,0, -0.1775,0,0])
A3= np.array([0,1,0, -0.4945,0,0])
A4= np.array([0,1,0, -0.6965,0,0])
A_list= [A1,A2,A3,A4]

def build_A_all(A_list):
    n= len(A_list)
    Aall= np.zeros((6*n,n))
    for i in range(n):
        Aall[6*i:6*(i+1), i] = A_list[i]
    return Aall

A_all= build_A_all(A_list)   # shape (24,4)

# Inertias
inertia_mats= {
    'link1': np.array([[0.000279744834534,0,0],
                       [0,0.000265717763008,0],
                       [0,0,6.53151584738e-05]]),
    'link2': np.array([[0.00251484771035,0,0],
                       [0,0.00248474836108,0],
                       [0,0,9.19936757328e-05]]),
    'link3': np.array([[0.000791433503053,0,0],
                       [0,0.000768905501178,0],
                       [0,0,6.88531064581e-05]]),
    'link4': np.array([[0.00037242266488,0,0],
                       [0,0.000356178538461,0],
                       [0,0,4.96474819141e-05]])
}
masses= [0.190421352, 0.29302326, 0.21931466, 0.15813986]

def build_spatial_inertia(I_3x3, m):
    G= np.zeros((6,6))
    G[0:3,0:3]= I_3x3
    G[3:6,3:6]= m*np.eye(3)
    return G

G_list=[]
for i in range(4):
    link_name= f'link{i+1}'
    I_i= inertia_mats[link_name]
    m_i= masses[i]
    G_i= build_spatial_inertia(I_i,m_i)
    G_list.append(G_i)

def block_diag_6x6(blocks):
    n= len(blocks)
    out= np.zeros((6*n,6*n))
    for i in range(n):
        out[6*i:6*(i+1), 6*i:6*(i+1)] = blocks[i]
    return out

G_all= block_diag_6x6(G_list)   # shape (24,24)

##############################################################################
#  3) W(q), L(q)=(I - W)^-1
##############################################################################
def build_W(q):
    """
    W is 24x24, with Ad_{T_{i+1,i}} in the (i, i+1) block.
    T_{i+1,i}= exp(-A_{i+1}q_{i+1}) M_{i+1}.
    """
    n= len(A_list)
    W_= np.zeros((6*n,6*n))
    for i in range(n-1):
        qip1= q[i+1]
        Ai1= A_list[i+1]
        Mi1= M_list[i+1]
        T_i1= exp_se3(Ai1, -qip1)@ Mi1
        AdT= adjoint(T_i1)
        row= 6*i
        col= 6*(i+1)
        W_[row:row+6, col:col+6] = AdT
    return W_

def build_L(q):
    n= len(q)
    I_= np.eye(6*n)
    W_= build_W(q)
    return np.linalg.inv(I_ - W_)

##############################################################################
#  4) Closed-form M, c, g 
##############################################################################
def skew_6(x):
    """
    x= [w(3), v(3)] => 6x6 ad_x
    """
    w= x[:3]
    v= x[3:6]
    mat= np.zeros((6,6))
    mat[0:3,0:3]= skew_3(w)
    mat[3:6,3:6]= skew_3(w)
    mat[3:6,0:3]= skew_3(v)
    return mat

def block_diag_ad(vec_6n):
    """
    build block diag of ad_{...} from a 6n vector
    """
    n_ = vec_6n.size//6
    out= np.zeros((6*n_,6*n_))
    for i_ in range(n_):
        chunk= vec_6n[6*i_: 6*(i_+1)]
        out[6*i_:6*(i_+1), 6*i_:6*(i_+1)] = skew_6(chunk)
    return out

def closed_form_M(q):
    Lq= build_L(q)
    tmp= Lq@A_all
    Mq= A_all.T@ Lq.T@ G_all@ tmp
    return Mq

JOINT_LIMITS = [
    (-1.57,1.57),
    (-1.57,1.57),
    (-1.57,1.57),
    (-1.57,1.57)
]

def clamp_joints(q):
    for i in range(len(q)):
        lo,hi= JOINT_LIMITS[i]
        if q[i]<lo:
            q[i]= lo
        elif q[i]>hi:
            q[i]= hi
    return q

def closed_form_dynamics(q, qdot):
    """
    Return M(q), c(q,qdot), g(q).
    1) M(q)= A^T L^T G L A
    2) For gravity => transform [0,0,-9.81,0,0,0] by Ad_{T_{1,0}}
       and place in first block => v_base_dot_6n
    3) Evaluate c+g => qdot!=0 => tau_0
       Evaluate g => qdot=0 => tau_g
       => c= (c+g)-g
    """
    n= len(q)
    sixn= 6*n

    # 1) M(q)
    Mq= closed_form_M(q)

    # 2) build base accel with adjacency
    T_10= M_list[0]   # from base to link1 at q=0
    AdT_10= adjoint(T_10)
    raw_g= np.array([0,0,0.0,0,0,-9.81])
    g_first= AdT_10@ raw_g
    v_base_dot_6n= np.zeros(sixn)
    v_base_dot_6n[0:6]= g_first

    # also define v_base_6n=0
    v_base_6n= np.zeros(sixn)

    # 3) c+g => qdot !=0 => no q_ddot
    Lq= build_L(q)
    Aqd= A_all@ qdot
    v_all= Lq@ (Aqd + v_base_6n)
    W_= build_W(q)
    adAqd= block_diag_ad(Aqd)
    v_all_dot_0= Lq@ ( - adAqd@ W_@ v_all + v_base_dot_6n )
    ad_v_all= block_diag_ad(v_all)
    tmp= G_all@ v_all_dot_0 - ad_v_all.T@ G_all@ v_all
    F_all_0= Lq.T@ tmp
    c_plus_g= A_all.T@ F_all_0

    # 4) pure gravity => qdot=0 => no Coriolis
    qdot_zero= np.zeros_like(qdot)
    Aqd_zero= A_all@ qdot_zero  # zero
    v_all_zero= Lq@ (Aqd_zero + v_base_6n) # => 0
    v_all_dot_g= Lq@ v_base_dot_6n
    ad_v0= block_diag_ad(v_all_zero)
    tmp_g= G_all@ v_all_dot_g - ad_v0.T@ G_all@ v_all_zero
    F_all_g= Lq.T@ tmp_g
    g_= A_all.T@ F_all_g

    c_= c_plus_g - g_

    return Mq, c_, g_

def forward_dynamics(q, qdot, tau):
    Mq,c_,g_= closed_form_dynamics(q,qdot)
    rhs= tau - c_ - g_
    qddot= np.linalg.inv(Mq)@ rhs
    return qddot

def step_state(x, tau, dt):
    q    = x[:4].copy()
    qdot = x[4:8].copy()
    qddot= forward_dynamics(q, qdot, tau)
    q_next= q + dt*qdot
    qdot_next= qdot + dt*qddot
    q_next= clamp_joints(q_next)
    return np.concatenate([q_next, qdot_next])

def forward_kinematics_link_origins(q):
    """
    Return link0..link4 frames.  link0= base
    """
    T_list=[]
    T_base= np.eye(4)
    T_list.append(T_base)
    for i in range(4):
        Ai= A_list[i]
        Mi= M_list[i]
        T_i= exp_se3(Ai, -q[i])@ Mi
        T_base= T_base@ T_i
        T_list.append(T_base)
    return [Ti[0:3,3] for Ti in T_list]

def plot_frame(ax, T, length=0.05, label=None):
    O= T[0:3,3]
    R= T[0:3,0:3]
    x_axis= R@ np.array([length,0,0])
    y_axis= R@ np.array([0,length,0])
    z_axis= R@ np.array([0,0,length])
    ax.plot([O[0], O[0]+x_axis[0]],
            [O[1], O[1]+x_axis[1]],
            [O[2], O[2]+x_axis[2]], 'r-')
    ax.plot([O[0], O[0]+y_axis[0]],
            [O[1], O[1]+y_axis[1]],
            [O[2], O[2]+y_axis[2]], 'g-')
    ax.plot([O[0], O[0]+z_axis[0]],
            [O[1], O[1]+z_axis[1]],
            [O[2], O[2]+z_axis[2]], 'b-')
    if label:
        ax.text(O[0], O[1], O[2], label)

##############################################################################
#  5) Inverse Dynamics, Animations
##############################################################################
def inverse_dynamics(q, qdot, qddot):
    """
    tau= M(q)*qddot + c(q,qdot)+ g(q).
    """
    Mq,c_,g_= closed_form_dynamics(q,qdot)
    tau= Mq@ qddot + c_ + g_
    return tau
  

def animate_trajectory_random():
    """
    Just random small torque => watch motion
    """
    dt=0.01
    T_sim=2.0
    steps= int(T_sim/dt)
    fig= plt.figure()
    ax= fig.add_subplot(111,projection='3d')
    plt.ion()
    np.random.seed(2)
    x= np.zeros(8)
    ee_traj=[]
    for k in range(steps):
        t= k*dt
        ax.clear()
        # random torque => small
        tau= 0.02*(2.*np.random.rand(4)-1.)
        x= step_state(x,tau,dt)
        link_orig= forward_kinematics_link_origins(x[:4])
        ee= link_orig[-1]
        ee_traj.append(ee.copy())

        for i in range(len(link_orig)-1):
            st= link_orig[i]
            en= link_orig[i+1]
            ax.plot([st[0],en[0]],
                    [st[1],en[1]],
                    [st[2],en[2]], 'b-o', linewidth=2)

        arr= np.array(ee_traj)
        ax.plot(arr[:,0],arr[:,1],arr[:,2],'r-')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(0,1.0)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.draw()
        plt.pause(dt)
    print("Final state:", x)
    plt.ioff()
    plt.show()
    

if __name__=="__main__":
    # 2) Animate small random torques
    animate_trajectory_random()
