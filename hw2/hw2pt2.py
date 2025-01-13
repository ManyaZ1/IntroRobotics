from hw2pt1 import *

def plot_box(ax):
    # box
    length_cm = 0.40  # x
    width_cm = 0.20   # y
    height_cm = 0.20  # z

    # Translation vector to move the box to (x=30, y=40) for its edge at (0, 0, 0)
    box_translation = np.array([0.30, 0.40, 0.0])

    # Define the vertices
    vertices = np.array([[0, 0, 0],
                        [length_cm, 0, 0],
                        [length_cm, width_cm, 0],
                        [0, width_cm, 0],
                        [0, 0, height_cm],
                        [length_cm, 0, height_cm],
                        [length_cm, width_cm, height_cm],
                        [0, width_cm, height_cm]])

    # Define the six faces (by connecting the vertices)
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]]]  +box_translation# Left

    # Create the box using Poly3DCollection with red color and semi-transparent surface
    ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=1, edgecolors='k', alpha=0.3))

    # Create the cylinder (radius 5, height 40) in the center of the box
    theta = np.linspace(0, 2 * np.pi, 100)
    z_cylinder = np.linspace(0, height_cm, 100)
    x_cylinder = 0.05 * np.cos(theta)
    y_cylinder = 0.05 * np.sin(theta)

    x_cylinder += 0.50
    y_cylinder += 0.50

    # Plot the cylinder in the box (this gives the effect of a hole)
    ax.plot_surface(x_cylinder[None, :], y_cylinder[None, :], z_cylinder[:, None], color='0.3', alpha=1)
'''    theta = np.linspace(0, 2 * np.pi, 100)  # Angular values for circular cross-section
    x_cylinder_horizontal = np.linspace(0.57, 0.82, 100)  # Extent of cylinder along X in meters
    y_cylinder_horizontal = 0.025 * np.sin(theta)+0.095  # Radius in meters
    z_cylinder_horizontal = 0.025 * np.cos(theta)+0.025  # Radius in meters
    ax.plot_surface(x_cylinder_horizontal[:, None], y_cylinder_horizontal[None, :], #prosoxh sta none!
                z_cylinder_horizontal[None, :], color='orange', alpha=1)'''
def plot_cylinder_orange(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    z_cylinder = np.linspace(0, 0.25, 100)
    x_cylinder = 0.025 * np.cos(theta)
    y_cylinder = 0.025 * np.sin(theta)

    x_cylinder += 0.50
    y_cylinder += 0.50
    z_cylinder += 0.20
    # Plot the cylinder in the box (this gives the effect of a hole)
    ax.plot_surface(x_cylinder[None, :], y_cylinder[None, :], z_cylinder[:, None], color='orange', alpha=1)


def plot_robot(positions, final_pose, target_pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_box(ax)
    plot_cylinder_orange(ax)
   
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
    # Target pose for the end-effector to reach the box
    target_pose = np.array([
        [0.0, 0.0,  1.0, 0.475],
        [0.0, -1.0, 0.0,  0.50],
        [1.0, 0.0, 0.0,  0.325],
        [0.0, 0.0,  0.0,  1.0]])

    # Initial guess for joint angles
    initial_guess =  np.ones(7)*0.6
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
        print(f"Joint {i}: {pos}")
    print("Joint Angles:")
    for i, theta in enumerate(joint_angles):
        print(f"Joint theta{i}: {theta}")

