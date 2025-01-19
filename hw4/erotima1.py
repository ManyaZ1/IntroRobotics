import xml.etree.ElementTree as ET
import pandas as pd
from urdfpy import URDF
import numpy as np
import os
import matplotlib.pyplot as plt
def extract_joints_links(robot):
    joints = robot.joints
    links = robot.links

    # Initialize arrays for joint and link data
    link_origins = []
    link_orientations = []
    link_names = []
    body_info = []

    # Extract Link 0's origin from its visual element
    link_0_origin = np.eye(4)  # Default to identity if no origin is specified
    if links[0].visuals and links[0].visuals[0].origin is not None:
        link_0_origin = links[0].visuals[0].origin

    link_origins.append(link_0_origin[:3, 3])  # Add Link 0's global origin
    link_orientations.append(link_0_origin[:3, :3])
    link_names.append(links[0].name)

    # Handle Link 0 CoM
    if links[0].inertial and links[0].inertial.origin is not None:
        link_0_com = links[0].inertial.origin
    else:
        link_0_com = link_0_origin  # Default CoM to link frame origin if no inertial data

    body_info.append({
        "link_name": links[0].name,
        "CoM": link_0_com,
        "mass": links[0].inertial.mass if links[0].inertial else 0,
        "inertia": links[0].inertial.inertia if links[0].inertial else np.zeros((3, 3)),
    })

    # Extract CoM and other data from each link
    for link in links[1:]:  # Start from Link 1
        com = link.inertial.origin if link.inertial and link.inertial.origin is not None else np.eye(4)
        mass = link.inertial.mass if link.inertial else 0
        inertia = link.inertial.inertia if link.inertial else np.zeros((3, 3))
        body_info.append({
            "link_name": link.name,
            "CoM": com,
            "mass": mass,
            "inertia": inertia,
        })

    # Traverse through the joints to compute global transformations
    current_transform = link_0_origin
    for joint in joints:
        current_transform = current_transform @ joint.origin
        position = current_transform[:3, 3]
        orientation = current_transform[:3, :3]
        link_origins.append(position)
        link_orientations.append(orientation)
        link_names.append(joint.child)
    return link_origins, link_orientations, link_names, body_info, joints, link_0_origin
def calculate_screw_axis(joint, global_transform):
    # Extract rotation axis r in the local joint frame
    r = np.array(joint.axis)  # Rotation axis (unit vector)
    #print(r)
    #print(f"Rotation Axis for Joint {joint.name}: {r}")
    # Transform r to the global frame using the joint's rotation matrix
    rotation_matrix = global_transform[:3, :3]  
    r_global = rotation_matrix @ r  # Rotation axis in the global frame

    # Extract q: a point on the axis in the global frame
    q_global = global_transform[:3, 3]
    #print(f"q:{q_global}")
    # h = 0 for revolute joints
    h = 0  # Assuming revolute joints

    # Compute screw axis
    screw_axis = np.hstack((r_global, np.cross(-r_global, q_global) + h * r_global))
    return screw_axis

def extract_screwaxis(joints, link_0_origin):
    screw_axes = []
    current_transform = np.eye(4)#link_0_origin # Initialize the global transformation
    #print(link_0_origin)  #4x4 matrix

    for joint in joints:
        # Update the global transformation for the current joint
        current_transform = current_transform @ joint.origin

        # Calculate the screw axis
        screw_axis = calculate_screw_axis(joint, current_transform)
        screw_axes.append(screw_axis)

        # Print screw axis for each joint
        #print(f"Screw Axis for Joint {joint.name}: {screw_axis}")
    return screw_axes
# Function to parse the URDF and extract kinematic and dynamic info
def parse_urdf(file_path):
    # Parse URDF
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    links_data = []
    joints_data = []

    # Extract links
    for link in root.findall('link'):
        name = link.attrib.get('name', 'unknown')
        inertial = link.find('inertial')
        
        mass, inertia = None, {}
        com_frame = [0, 0, 0]
        
        if inertial is not None:
            mass_element = inertial.find('mass')
            if mass_element is not None:
                mass = float(mass_element.attrib.get('value', 0))
            
            inertia_element = inertial.find('inertia')
            if inertia_element is not None:
                inertia = {k: float(inertia_element.attrib.get(k, 0)) for k in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']}
            
            origin = inertial.find('origin')
            if origin is not None:
                com_frame = [float(x) for x in origin.attrib.get('xyz', '0 0 0').split()]

        links_data.append({
            'name': name,
            'mass': mass,
            'inertia': inertia,
            'center_of_mass': com_frame
        })

    # Extract joints
    for joint in root.findall('joint'):
        name = joint.attrib.get('name', 'unknown')
        joint_type = joint.attrib.get('type', 'unknown')
        
        origin = joint.find('origin')
        joint_frame = [float(x) for x in origin.attrib.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
        
        axis = joint.find('axis')
        screw_axis = [float(x) for x in axis.attrib.get('xyz', '0 0 0').split()] if axis is not None else [0, 0, 0]
        
        parent = joint.find('parent')
        child = joint.find('child')
        parent_link = parent.attrib.get('link', 'unknown') if parent is not None else 'unknown'
        child_link = child.attrib.get('link', 'unknown') if child is not None else 'unknown'
        
        joints_data.append({
            'name': name,
            'type': joint_type,
            'parent_link': parent_link,
            'child_link': child_link,
            'joint_frame': joint_frame,
            'rotation_axis': screw_axis
        })

    return links_data, joints_data

# Add the visualization for screw axes (if needed)
def plot_frame(ax, T, length=0.05):
    origin = T[:3, 3]
    R = T[:3, :3]
    xaxis = R @ np.array([1., 0., 0.])
    yaxis = R @ np.array([0., 1., 0.])
    zaxis = R @ np.array([0., 0., 1.])

    ax.quiver(*origin, *xaxis, color='r', length=length)
    ax.quiver(*origin, *yaxis, color='g', length=length)
    ax.quiver(*origin, *zaxis, color='b', length=length)
def plot_screw_axes(ax, screw_axes, link_origins):
    for i, (screw_axis, origin) in enumerate(zip(screw_axes, link_origins)):
        r = screw_axis[:3]  # Rotation axis
        q = origin  # A point on the screw axis

        # Draw the rotation axis
        ax.quiver(*q, *r, color='orange', length=0.1, label=f"Screw Axis {i+1}" if i == 0 else "")

def visualize_robot(link_origins, link_orientations, link_names, body_info, joints):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot links as blue sticks
    for i in range(len(link_origins) - 1):
        start = link_origins[i]
        end = link_origins[i + 1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color='blue', linewidth=2, label="Link" if i == 0 else ""
        )

    # Plot joints as blue dots
    for origin, name in zip(link_origins, link_names):
        ax.scatter(*origin, color='blue', s=50, label="Joint" if name == link_names[0] else "")

    # Initialize transformation for the base frame
    current_transform = link_0_origin

    # Plot CoM frames and body frames
    for i, body in enumerate(body_info):
        com_local = body["CoM"][:3, 3] if body["CoM"].shape == (4, 4) else np.zeros(3)

        # Transform CoM to global coordinates using the link frame (not the joint frame)
        com_global = current_transform @ body["CoM"]

        # Extract the CoM position in global coordinates
        com_global_position = com_global[:3, 3]

        # Print the global coordinates of the CoM
        #print(f"Global CoM for {body['link_name']}: {com_global_position}")

        ax.scatter(*com_global_position, color="purple", marker="x", s=50, label="CoM" if i == 0 else "")
        ax.text(*com_global_position, f"CoM: {body['link_name']}", fontsize=8)

        # Update the transformation for the joint if it exists
        if i < len(joints):
            current_transform = current_transform @ joints[i].origin

        # Plot the body frame at the global transformation of the link
        plot_frame(ax, current_transform, length=0.05)

    # Set plot limits and labels
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Robot Visualization with Links, Joints, and CoM")

    plt.legend(loc="upper left")
    plt.show()

# Include screw axes in the visualization
def visualize_robot_with_screw_axes(link_origins, link_orientations, link_names, body_info, joints, screw_axes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Existing visualization
    for i in range(len(link_origins) - 1):
        start = link_origins[i]
        end = link_origins[i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=2)

    # Plot screw axes
    plot_screw_axes(ax, screw_axes, link_origins[:-1])

    # Set limits and labels
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Robot with Rotation Axes")

    plt.legend(loc="upper left")
    plt.show()



def calculate_home_configuration(robot):
    """
    Calculate the home configuration of the robot.
    This function computes the global transformation matrices of all links in their default positions.
    """
    joints = robot.joints
    links = robot.links

    # Initialize home configuration array
    home_configuration = []
    current_transform = np.eye(4)  # Start with identity matrix (base link frame)

    for joint in joints:
        # Update the global transform using the joint's origin (home position)
        current_transform = current_transform @ joint.origin
        print("##################")
        print(joint.origin)
        home_configuration.append(current_transform.copy())  # Store the transformation matrix

    return home_configuration
def calculate_end_effector_home(robot):
    """
    Calculate the end effector's position and orientation (M) in the zero configuration.
    """
    joints = robot.joints
    current_transform = np.eye(4)  # Identity matrix as the starting transform

    # Multiply all joint transforms to calculate the end effector transform
    for joint in joints:
        current_transform = current_transform @ joint.origin

    return current_transform
# Visualize the robot with screw axes
#visualize_robot_with_screw_axes(link_origins, link_orientations, link_names, body_info, joints, screw_axes)

# Main execution
if __name__ == "__main__":
    # File path to the URDF file
    currentpath = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(currentpath, 'arm.urdf')
    
    # Parse URDF and display data
    links_data, joints_data = parse_urdf(file_path)
    
    print("Links Data:")
    for link in links_data:
        print(link)
    
    print("\nJoints Data:")
    for joint in joints_data:
        print(joint)
    
    # # Visualization with screw axes
    robot = URDF.load(file_path)
    link_origins, link_orientations, link_names, body_info, joints, link_0_origin = extract_joints_links(robot)
    screw_axes = extract_screwaxis(joints, link_0_origin)
    visualize_robot(link_origins, link_orientations, link_names, body_info, joints)
    visualize_robot_with_screw_axes(link_origins, link_orientations, link_names, body_info, joints, screw_axes)
    screw_axes = extract_screwaxis(joints, link_0_origin)
    print("screw axes")
    for arra in screw_axes:
        print(arra)
    #print(screw_axes)

