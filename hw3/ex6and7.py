import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define skew-symmetric matrix function
def skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

# Product of Exponentials Forward Kinematics
def poe_forward_kinematics(screw_axes, joint_angles, home_config):
    T = np.eye(4)
    positions = [T[:3, 3].tolist()]  # Start with base origin

    for i in range(len(joint_angles)):
        twist_hat = np.zeros((4, 4))
        twist_hat[:3, :3] = skew(screw_axes[:3, i])
        twist_hat[:3, 3] = screw_axes[3:, i]
        exp_twist = expm(twist_hat * joint_angles[i])
        T = T @ exp_twist
        T2 = T @ home_config
        positions.append(T2[:3, 3].tolist())

    end_effector_pose = T @ home_config
    return end_effector_pose, positions

# Cubic Spline Trajectory in SE(3)
def cubic_spline_se3(T_s, T_g, T_h, T, num_points):
    tau_s = logm(T_s)
    tau_g = logm(T_g)
    #tau_h = logm(T_h)
    
    c0 = tau_s
    c1 = np.zeros((4, 4))
    c2 = (3 / T**2) * (tau_g - tau_s) - (2 / T) * np.zeros((4,4)) - (1 / T) * np.zeros((4,4))
    c3 = (-2 / T**3) * (tau_g - tau_s) + (1 / T**2) * (np.zeros((4,4)) + np.zeros((4,4)))
    
    time_points = np.linspace(0, T, num_points)
    positions = []
    rotations = []
    transformations = []
    
    for t in time_points:
        tau_t = c0 + c1 * t + c2 * t**2 + c3 * t**3
        T_t = expm(tau_t)
        
        T_t = np.real(T_t)  # Ensure real numbers
        transformations.append(T_t)
        positions.append(T_t[:3, 3])
        rotations.append(T_t[:3, :3])
    #print(positions)
    return np.array(positions), np.array(rotations), transformations

# Inverse Kinematics Solver using Proportional Control
def inverse_kinematics_control(screw_axes, T0, target_pose, initial_guess, max_steps=1000, tolerance=1e-6):
    joint_angles = initial_guess.copy()
    position_gain = 5.0
    orientation_gain = 5.0
    dt = 0.1  # Time step for simulation

    for step in range(max_steps):
        T_current, _ = poe_forward_kinematics(screw_axes, joint_angles, T0)
        position_current = T_current[:3, 3]
        orientation_current = T_current[:3, :3]

        # Compute position and orientation errors
        e_p = target_pose[:3, 3] - position_current
        R_error = target_pose[:3, :3] @ orientation_current.T
        log_R = logm(R_error)
        log_R = np.real(log_R)  # Discard negligible imaginary parts
        e_o = np.array([log_R[2,1], log_R[0,2], log_R[1,0]])

        # Combine errors with gains
        e = np.concatenate((position_gain * e_p, orientation_gain * e_o))

        # Check for convergence
        if np.linalg.norm(e_p) < tolerance and np.linalg.norm(e_o) < tolerance:
            print(f"Converged in {step} steps.")
            break

        # Compute Jacobian numerically
        J = compute_jacobian(screw_axes, joint_angles, T0)

        # Damped Least Squares for joint velocities
        damping_factor = 1e-3
        JTJ = J.T @ J + damping_factor * np.eye(J.shape[1]) # λ=1e-3
        try:
            J_damped = np.linalg.inv(JTJ) @ J.T
        except np.linalg.LinAlgError:
            print(f"Singular matrix encountered at step {step}.")
            return joint_angles
        joint_velocities = J_damped @ e  # v=j*e

        # Ensure joint_velocities are real
        joint_velocities = np.real(joint_velocities)

        # Update joint angles
        joint_angles += joint_velocities * dt

    else:
        print("Inverse kinematics did not converge within the maximum number of steps.")

    return joint_angles

# Compute Jacobian Numerically
def compute_jacobian(screw_axes, joint_angles, T0):
    n_joints = screw_axes.shape[1]
    J = np.zeros((6, n_joints))
    delta = 1e-6

    T_current, _ = poe_forward_kinematics(screw_axes, joint_angles, T0)
    p_current = T_current[:3, 3]
    R_current = T_current[:3, :3]

    for i in range(n_joints):
        perturbed_angles = joint_angles.copy()
        perturbed_angles[i] += delta
        T_perturbed, _ = poe_forward_kinematics(screw_axes, perturbed_angles, T0)

        p_perturbed = T_perturbed[:3, 3]
        R_perturbed = T_perturbed[:3, :3]

        dp = (p_perturbed - p_current) / delta
        R_error = R_perturbed @ R_current.T
        log_R = logm(R_error)
        log_R = np.real(log_R)  # Discard negligible imaginary parts
        euler = np.array([log_R[2,1], log_R[0,2], log_R[1,0]]) / delta

        J[:, i] = np.concatenate((dp, euler))

    return J

# Visualization Functions
def visualize_robot(ax, positions, object_pose=None):
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', label='Robot Links', color='blue')
    
    # Annotate joints
    for i, (x, y, z) in enumerate(positions):
        ax.text(x, y, z, f'J{i}', color='red')

    if object_pose is not None:
        ax.scatter(object_pose[0, 3], object_pose[1, 3], object_pose[2, 3], color='magenta', s=100, label='Object')

    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([0, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Robot Visualization")

def draw_box(ax, center, size):
    x, y, z = center
    dx, dy, dz = size

    # Define the vertices of the box
    corners = np.array([
        [x - dx/2, y - dy/2, z - dz/2],
        [x + dx/2, y - dy/2, z - dz/2],
        [x + dx/2, y + dy/2, z - dz/2],
        [x - dx/2, y + dy/2, z - dz/2],
        [x - dx/2, y - dy/2, z + dz/2],
        [x + dx/2, y - dy/2, z + dz/2],
        [x + dx/2, y + dy/2, z + dz/2],
        [x - dx/2, y + dy/2, z + dz/2],
    ])

    # Define the 12 edges of the box
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    for edge in edges:
        points = corners[list(edge)]
        ax.plot(points[:,0], points[:,1], points[:,2], color='cyan')

# Grasp and Release Conditions
def can_grasp(p_effector, R_effector, p_target, R_target, grasp_threshold=1e-2, orientation_threshold=1e-2):
    # Έλεγχος θέσης (ίδιος με πριν)
    position_error = np.linalg.norm(p_effector - p_target)
    
    # Έλεγχος προσανατολισμού
    R_error = R_target @ R_effector.T  # Σχετική περιστροφή μεταξύ στόχου και end-effector
    log_R = logm(R_error)  # Υπολογισμός λογαρίθμου του σφάλματος περιστροφής
    orientation_error = np.linalg.norm(log_R)  # Υπολογισμός norm του log_R ως μέτρο σφάλματος
    
    return position_error < grasp_threshold and orientation_error < orientation_threshold
def can_release(p_effector, R_effector, p_target, R_target, release_threshold=1e-2, orientation_threshold=1e-2):
    # Έλεγχος θέσης (ίδιος με πριν)
    position_error = np.linalg.norm(p_effector - p_target)
    
    # Έλεγχος προσανατολισμού
    R_error = R_target @ R_effector.T
    log_R = logm(R_error)
    orientation_error = np.linalg.norm(log_R)
    
    return position_error < release_threshold and orientation_error < orientation_threshold

# Kinematic Simulation with Trajectory and Visualization
def kinematic_simulation_with_trajectory(screw_axes, T0, positions, rotations, grasp_pose, release_pose, object_pose):
    time_step = 0.1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    # Initialize box parameters (optional)
    box_position = [0.5, 0.5, 0.0]
    box_size = [0.4, 0.2, 0.2]
    # draw_box(ax, box_position, box_size)

    joint_angles = np.zeros(screw_axes.shape[1])
    grasped = False
    released = False
    release_complete = False

    for i, (position, rotation) in enumerate(zip(positions, rotations)):
        T_current, joint_positions = poe_forward_kinematics(screw_axes, joint_angles, T0)
        current_position = T_current[:3, 3]
        current_rotation = T_current[:3, :3]

        # Calculate errors
        error_position = position - current_position
        R_error = rotation @ current_rotation.T
        orientation_error_matrix = logm(R_error)
        orientation_error_matrix = np.real(orientation_error_matrix)  # Ensure real
        error_orientation = np.array([orientation_error_matrix[2, 1], orientation_error_matrix[0, 2], orientation_error_matrix[1, 0]])

        # Apply control gains
        position_gain = 5.0
        orientation_gain = 5.0
        error = np.concatenate((position_gain * error_position, orientation_gain * error_orientation))

        # Compute Jacobian
        J = compute_jacobian(screw_axes, joint_angles, T0)

        # Compute joint velocities using damped least squares
        damping_factor = 1e-3
        JTJ = J.T @ J + damping_factor * np.eye(J.shape[1])
        try:
            J_damped = np.linalg.inv(JTJ) @ J.T
        except np.linalg.LinAlgError:
            print(f"Singular matrix encountered at step {i}. Skipping update.")
            J_damped = np.zeros((J.shape[1], J.shape[0]))  # Adjusted to match dimensions

        joint_velocities = J_damped @ error
        joint_velocities = np.real(joint_velocities)  # Ensure real

        # Update joint angles
        joint_angles += joint_velocities * time_step

        # Update visualization
        ax.clear()
        # draw_box(ax, box_position, box_size)  # Optional
        visualize_robot(ax, joint_positions, object_pose)

        if not grasped and can_grasp(current_position, current_rotation, grasp_pose, target_pose1[:3, :3]):
            print(f"Grasped object at step {i}.")
            print(T_current)
            grasped = True
            object_pose[:3, 3] = grasp_pose  # Attach object to end-effector
            object_pose[:3, :3] = target_pose1[:3, :3]  # Align rotation

        if grasped and not released and can_release(current_position, current_rotation, release_pose, target_pose2[:3, :3]):
            print(f"Released object at step {i}.")
            print(T_current)
            released = True
            object_pose = np.eye(4)
            object_pose[:3, 3] = release_pose
            object_pose[:3, :3] = target_pose2[:3, :3]  # Align final orientation

        # Scatter grasp and release positions
        ax.scatter(grasp_pose[0], grasp_pose[1], grasp_pose[2], color='green', s=100, label='Grasp Position')
        ax.scatter(release_pose[0], release_pose[1], release_pose[2], color='red', s=100, label='Release Position')

        # Scatter the object
        if grasped and not released:
            ax.scatter(object_pose[0, 3], object_pose[1, 3], object_pose[2, 3], color='magenta', s=100, label='Object')
        elif released:
            ax.scatter(object_pose[0, 3], object_pose[1, 3], object_pose[2, 3], color='yellow', s=100, label='Released Object')
            plt.ioff()
            plt.show()
            return joint_angles

        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-0.5, 1.5])
        ax.set_zlim([0, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("Kinematic Simulation with Grasp and Release")

        plt.draw()
        plt.pause(time_step)

    plt.ioff()
    plt.show()
    return joint_angles
'''
        # Check for grasp
        if not grasped and can_grasp(current_position, grasp_pose):
            print(f"Grasped object at step {i}.")
            print(T_current)
            grasped = True
            object_pose[:3, 3] = grasp_pose  # Attach object to end-effector

        # Check for release
        if grasped and not released and can_release(current_position, release_pose):
            print(f"Released object at step {i}.")
            print(T_current)
            released = True
            # Detach object from end-effector (object remains at release_pose)
            object_pose = np.eye(4)
            object_pose[:3, 3] = release_pose

        # Update object position if grasped
        if grasped and not released:
            object_pose[:3, 3] = current_position
'''
# Main Execution Block
if __name__ == "__main__":
    # Define the screw axes for the robot's joints (6x7 matrix)
    screw_axes = np.array([
        [0, 0, 1, 0, 0, 0], 
        [0, 1, 0, -0.333, 0, 0], 
        [0, 0, 1, 0, 0, 0],
        [0, -1, 0, 0.649, 0, -0.088], 
        [0, 0, 1, 0, 0, 0], 
        [0, -1, 0, 1.033, 0, 0],
        [0, 0, 1, 0, -0.088, 0]
    ]).T  # Transpose to make it 6x7

    # Home configuration of the end-effector
    T0 = np.array([
        [1, 0, 0, 0.088], 
        [0, -1, 0, 0], 
        [0, 0, -1, 0.926], 
        [0, 0, 0, 1]
    ]) 

    # Define the start, first target, and second target poses
    T_s = np.array([
        [1, 0, 0, 0.088],
        [0, -1, 0, 0],
        [0, 0, -1, 0.926],
        [0, 0, 0, 1]
    ])

    target_pose1 = np.array([
        [1, 0, 0, 0.65],
        [0, -1, 0, 0.095],
        [0, 0, -1, 0.05],
        [0, 0, 0, 1]
    ])

    target_pose2 = np.array([
        [0, 0, 1, 0.5],
        [0, -1, 0, 0.5],
        [1, 0, 0, 0.5],
        [0, 0, 0, 1]
    ])

    # Initial guess for inverse kinematics
    initial_guess = np.ones(screw_axes.shape[1]) * 0.6

    # Solve inverse kinematics for the first target pose
    joint_angles_initial = inverse_kinematics_control(screw_axes, T0, target_pose1, initial_guess)

    # Generate cubic spline trajectories between poses
    positions1, rotations1, transformations1 = cubic_spline_se3(T_s, target_pose1, target_pose1, T=10, num_points=100)
    positions2, rotations2, transformations2 = cubic_spline_se3(target_pose1, target_pose2, target_pose2, T=10, num_points=100)

    # Combine trajectory segments
    positions = np.vstack((positions1, positions2))
    rotations = np.vstack((rotations1, rotations2))

    # Define grasp and release poses
    grasp_pose = target_pose1[:3, 3]
    release_pose = target_pose2[:3, 3]

    # Initialize object pose
    object_pose = np.eye(4)
    object_pose[:3, 3] = grasp_pose  # Initially at grasp pose

    # Run the kinematic simulation with visualization
    final_joint_angles = kinematic_simulation_with_trajectory(
        screw_axes=screw_axes,
        T0=T0,
        positions=positions,
        rotations=rotations,
        grasp_pose=grasp_pose,
        release_pose=release_pose,
        object_pose=object_pose
    )

    # Compute and display the final pose from forward kinematics
    final_pose, joint_positions = poe_forward_kinematics(screw_axes, final_joint_angles, T0)
    #print("Final Pose from Forward Kinematics:\n", final_pose)
