from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.linalg import expm, logm  # logm for matrix logarithm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hw1script import RotX, RotY, RotZ, homogeneous, plot_frame
#1. Define the Robot's Screw Axes
#skew symmetric lecture
def hat(vec):
    v = vec.reshape((3,))
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])
#exp_rotation lecture
#ef exp_rotation(p): 
    #phi = p.reshape((3, 1))
    #theta = np.linalg.norm(phi)
    #if theta < 1e-12:
    #    return np.eye(3, 3)
    #a = phi / theta
    #return np.eye(3) * np.cos(theta) + (1. - np.cos(theta)) * a @ a.T + np.sin(theta) * hat(a)
#3. Solve Inverse Kinematics Using Newton-Raphson
def poe_forward_kinematics(screw_axes, joint_angles,M):
    """
    computes the pose of the robot's end-effector in the base frame given the joint angles and the robot's screw
    axes. It is based on the Product of Exponentials (PoE) formula, a standard approach in robotics for calculating forward kinematics.
    Compute the forward kinematics using the Product of Exponentials (PoE) formula.
    Parameters:
        screw_axes (numpy.ndarray): A 6xN matrix, where each column represents a screw axis in space frame.
        joint_angles (numpy.ndarray): A 1D array of joint angles (length N).
        M (numpy.ndarray): A 4x4 matrix representing the home configuration of the end-effector.
    Returns:
        end_effector_position (numpy.ndarray): The 4x4 transformation matrix of the end-effector in the base frame.
        positions (list): A list of joint positions (3D vectors) as computed along the way.
    """
    # Initialize the transformation matrix to identity (base frame to base frame)
    T = np.eye(4)
    # List to store joint positions, starting from the base frame origin
    positions = [[0, 0, 0]]  # The base frame origin is assumed to be [0, 0, 0]
    for i in range(len(joint_angles)):
        # Construct the twist matrix (se(3)) from the screw axis
        w_hat = np.zeros((4, 4))
        w_hat[:3, :3] = hat(screw_axes[:3, i])  # Angular velocity (skew-symmetric)
        w_hat[:3, 3] = screw_axes[3:, i]       # Linear velocity

        # Compute the exponential map (transformation matrix for this joint)
        exp_twist = expm(w_hat * joint_angles[i])#e^sθ

        # Update the overall transformation matrix
        T = T @ exp_twist

        # Compute the current position of the joint relative to the base frame
        T_current = T @ M
        positions.append(T_current[:3, 3])  # Extract the position vector

    # Compute the final end-effector transformation relative to the base frame
    end_effector_position = T @ M
    return end_effector_position, positions

# Inverse kinematics using Newton-Raphson method
def solve_inverse_kinematics(screw_axes, M, target_pose, initial_angles, max_iterations=1000, tolerance=1e-4):
    """
    Solves the inverse kinematics problem using the Newton-Raphson method.

    Parameters:
        screw_axes (numpy.ndarray): A 6xN matrix of screw axes for the robot's joints.
        M (numpy.ndarray): The 4x4 home configuration matrix of the end-effector.
        target_pose (numpy.ndarray): The 4x4 target pose matrix of the end-effector.
        initial_angles (numpy.ndarray): Initial guess for the joint angles (1D array).
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Error norm tolerance for convergence.

    Returns:
        joint_angles (numpy.ndarray): The computed joint angles that achieve the desired pose.
        success (bool): Whether the algorithm converged to a solution.
    """
    joint_angles = np.array(initial_angles, dtype=np.float64)
    error_history = []  # Store error norms for debugging

    for iteration in range(max_iterations):
        # Compute the current end-effector pose and joint positions
        current_pose, _ = poe_forward_kinematics(screw_axes, joint_angles, M)

        # Decompose the target and current poses
        target_position = target_pose[:3, 3]
        target_orientation = target_pose[:3, :3]

        current_position = current_pose[:3, 3]
        current_orientation = current_pose[:3, :3]

        # Compute position error
        position_error = target_position - current_position

        # Compute orientation error using matrix logarithm
        R_rel = target_orientation @ current_orientation.T  # Relative rotation matrix
        orientation_error_matrix = logm(R_rel)  # Matrix logarithm for rotation error
        orientation_error = np.array([
            orientation_error_matrix[2, 1],
            orientation_error_matrix[0, 2],
            orientation_error_matrix[1, 0]
        ])

        # Combine position and orientation errors
        error_vector = np.concatenate((position_error, orientation_error))

        # Check convergence (if error norm is below the threshold)
        error_norm = np.linalg.norm(error_vector)
        error_history.append(error_norm)
        print(f'iteration {iteration} error norm = {error_norm:.4f}')
        if error_norm < tolerance:
            print(f"Converged after {iteration} iterations.")
            # Plot convergence history
            plt.plot(error_history)
            plt.xlabel("Iteration")
            plt.ylabel("Error Norm")
            plt.title("Convergence of Newton-Raphson Inverse Kinematics")
            plt.grid(True)
            plt.show()
            return joint_angles

        # Compute the Jacobian numerically
        num_joints = len(joint_angles)
        J = np.zeros((6, num_joints))
        delta = 1e-6  # Small perturbation for numerical differentiation
        for i in range(num_joints):
            # Perturb one joint angle at a time
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += delta
            perturbed_pose, _ = poe_forward_kinematics(screw_axes, perturbed_angles, M)

            # Compute the position perturbation
            perturbed_position = perturbed_pose[:3, 3]
            position_perturbation = perturbed_position - current_position

            # Compute the orientation perturbation
            perturbed_orientation = perturbed_pose[:3, :3]
            R_pert_rel = perturbed_orientation @ current_orientation.T
            rotation_perturbation_matrix = logm(R_pert_rel)
            orientation_perturbation = np.array([
                rotation_perturbation_matrix[2, 1],
                rotation_perturbation_matrix[0, 2],
                rotation_perturbation_matrix[1, 0]
            ])

            # Update the Jacobian column
            J[:, i] = np.concatenate((position_perturbation, orientation_perturbation)) / delta

        # Compute the pseudo-inverse of the Jacobian
        damping_factor = 0.01  # Regularization factor for stability
        J_damped = np.linalg.inv(J.T @ J + damping_factor * np.eye(num_joints)) @ J.T

        # Update joint angles using the Newton-Raphson method
        delta_angles = J_damped @ error_vector
        joint_angles += np.real(delta_angles)  # Ensure the update is real

    # If convergence is not achieved within max_iterations
    print(f"Failed to converge after {max_iterations} iterations.")
    #return joint_angles, False
    

def plot_robot(positions, final_pose, target_pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #horizontal cylinder
    # Horizontal cylinder
    theta = np.linspace(0, 2 * np.pi, 100)  # Angular values for circular cross-section
    x_cylinder_horizontal = np.linspace(0.57, 0.82, 100)  # Extent of cylinder along X in meters
    y_cylinder_horizontal = 0.025 * np.sin(theta)+0.095  # Radius in meters
    z_cylinder_horizontal = 0.025 * np.cos(theta)+0.025  # Radius in meters
    ax.plot_surface(x_cylinder_horizontal[:, None], y_cylinder_horizontal[None, :], #prosoxh sta none!
                z_cylinder_horizontal[None, :], color='orange', alpha=1)
    # Create a mesh grid for the cylinder surface
    #X, T = np.meshgrid(x_cylinder_horizontal, theta)  # Mesh grid for X and angular theta
    #Y = 0.025 * np.sin(T) + 0.095  # Adjust Y for offset, all in meters
    #Z = 0.025 * np.cos(T) + 0.025  # Adjust Z for offset, all in meters

    # Plot the cylinder surface
    #ax.plot_surface(X, Y, Z,color='orange', alpha=0.7, edgecolor='none' )
    
    # Extract x, y, z coordinates of joint positions
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    z = [p[2] for p in positions]

    # Plot the robot links
    ax.plot(x, y, z, marker='o', color='b', label='Robot Links')

    # Annotate joint positions
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi, yi, zi, f'J{i}', color='red')

    # Plot the target pose
    target_x, target_y, target_z = target_pose[:3, 3]
    ax.scatter(target_x, target_y, target_z, color='green', s=100, label='Target Pose')

    # Plot the end-effector position
    final_x, final_y, final_z = final_pose[:3, 3]
    ax.scatter(final_x, final_y, final_z, color='orange', s=100, label='End-Effector')
    
    # Connect target pose to the end-effector
    ax.plot(
        [final_x, target_x],
        [final_y, target_y],
        [final_z, target_z],
        'r--', label='Alignment Error'
    )

    # Set axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Visualization with Target and End-Effector")
    plt.legend()

    # Enforce equal aspect ratio for better visualization
    max_range = np.array([x, y, z]).ptp(axis=1).max() / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    # Define screw axes for a 7-DOF manipulator
    screw_axes = np.array([
        [0, 0, 1, 0, 0, 0],       # Joint 1 &
        [0, 1, 0, -0.333, 0, 0],   # Joint 2 &
        [0, 0, 1, 0, 0, 0],  # Joint 3 &
        [0, -1, 0, 0.649, 0,-0.088],   # Joint 4&
        [0, 0, 1, 0, 0, 0],  # Joint 5
        [0, -1, 0, 1.033, 0, 0],   # Joint 6
        [0, 0, -1, 0, 0.088, 0]   # Joint 7
    ]).T
    # Home configuration matrix (M)
    M = np.array([
        [1, 0, 0, 0.088],
        [0, -1, 0, 0],
        [0, 0, -1, 0.926],
        [0, 0, 0, 1]
    ]) 
    # Target pose for the end-effector to grasp the cylinder
    target_pose = np.array([
        [1.0, 0.0,  0.0, 0.695],
        [0.0, -1.0, 0.0,  0.0950],
        [0.0, 0.0, -1.0,  0.05],
        [0.0, 0.0,  0.0,  1.0]])
    # Initial guess for joint angles
    initial_guess = np.ones(7)*0.6
    # Perform inverse kinematics
    joint_angles =  solve_inverse_kinematics(screw_axes, M, target_pose, initial_guess)

    # Visualize the robot and target pose
    final_pose, positions = poe_forward_kinematics(screw_axes, joint_angles, M)
    # Simulated final pose (not accurate for actual simulation)
    plot_robot(positions, final_pose, target_pose)
    print("Final Pose from Forward Kinematics:\n", final_pose)
    print(positions)
        # Check error between target and final pose
    pose_error = np.linalg.norm(target_pose - final_pose)
    print(f"Pose error after solving IK: {pose_error}")
    print("Joint Position:")
    for i, pos in enumerate(positions):
        print(f"Joint{i}: {pos}")
    print("Joint Angles:")
    for i, theta in enumerate(joint_angles):
        print(f"Joint theta{i}: {theta}")

