import numpy as np
import cv2
from skspatial.objects import Plane, Points
from scipy.spatial.transform import Rotation
import open3d as o3d
from typing import Optional

def get_hand_pose_world_frame(hand_landmarks_world):
    """ Get pose of hand reference frame in world frame """

    # Fit plane to palm points
    palm_points = hand_landmarks_world[(0, 1, 5, 9, 13, 17), :]
    points = Points(palm_points)
    plane = Plane.best_fit(points)


    # Origin of hand ref frame, in world frame
    #origin = plane.point # Centroid of palm plane
    origin = hand_landmarks_world[9] # Base of middle finger

    # Get orientation, for hand to world frame
    # Define x-axis of palm reference frame
    # to be normal of palm plane
    x_vec = plane.normal
    x_axis = -x_vec / np.linalg.norm(x_vec)
    if np.dot(x_axis, np.array([0.0, 0.0, 1.0])) > 0.0:
        x_axis = -x_axis
    # Define z-axis of ref frame to be pointing from wrist to base of middle finger
    z_vec = -1 * plane.project_point(hand_landmarks_world[0, :] - origin)
    #z_vec = plane.project_point(hand_landmarks_world[9, :] - origin)
    z_axis = z_vec / np.linalg.norm(z_vec)
    # Define y-axis
    y_vec = np.cross(z_axis, x_axis)
    y_axis = y_vec / np.linalg.norm(y_vec)
    rotation_matrix_hand_to_world = np.vstack((x_axis, y_axis, z_axis))

    rotation_matrix_world_to_hand = np.linalg.inv(rotation_matrix_hand_to_world)

    hand_orientation = Rotation.from_matrix(rotation_matrix_world_to_hand)

    return origin, hand_orientation.as_matrix()

def get_ftip_pos_from_landmarks(landmarks):
    """ Get fingertip positions from hand landmarks [21, 3] """

    return landmarks[(4, 8, 12, 16, 20), :]

def get_allegro_ftip_pos_from_landmarks(landmarks):
    """ Get fingertip positions from hand landmarks [21, 3] """

    return landmarks[(8, 12, 16, 4), :]


def get_H_hand_to_world(hand_landmarks_world):

    hand_pos_wf, hand_R_wf = get_hand_pose_world_frame(hand_landmarks_world)

    hand_transform = np.eye(4)
    hand_transform[:3, :3] = hand_R_wf
    hand_transform[:3, 3] = hand_pos_wf

    return hand_transform

def get_H_inv(H):
    """Get inverse of homogenous transformation matrix H"""

    H_inv = np.zeros(H.shape)
    H_inv[3, 3] = 1
    R = H[:3, :3]
    P = H[:3, 3]

    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ P

    return H_inv

def transform_pts(pts_in, H):
    """Transform pts ([N, 3] or [3,]) by H"""
    # Check pts dim and reshape
    if pts_in.ndim == 1:
        pts = np.expand_dims(pts_in, axis=0)
    else:
        pts = pts_in

    pts_new = np.concatenate([pts, np.ones_like(pts[:, :1])], axis=-1)  # [N, 4]
    pts_new = np.matmul(H, pts_new.T).T[
        :, :3
    ]  # [4, 4], [4, N] -> [4, N] -> [N, 4] -> [N, 3]

    if pts_in.ndim == 1:
        return np.squeeze(pts_new)
    else:
        return pts_new
    

def save_as_pcd(pts, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Color points in point cloud
    palm_ids = [0, 1, 5, 9, 13, 17]
    thumb_ids = [4]
    index_ids = [8]
    mid_ids = [12]
    ring_ids = [16]
    pinky_ids = [20]
    #thumb_ids = [2, 3, 4]
    #index_ids = [6, 7, 8]
    #mid_ids = [10, 11, 12]
    #ring_ids = [14, 15, 16]
    #pinky_ids = [18, 19, 20]

    black = [0, 0, 0]
    red = [1, 0, 0]
    orange = [1, 0.486, 0]
    green = [0.02, 0.741, 0]
    blue = [0, 0.09, 0.741]
    purple = [0.663, 0, 0.82]
    grey = [0.6, 0.6, 0.6]

    colors = np.zeros(pts.shape)
    for i in range (pts.shape[0]):
        if i in palm_ids:
            c = black
        elif i in thumb_ids:
            c = red
        elif i in index_ids:
            c = orange
        elif i in mid_ids:
            c = green
        elif i in ring_ids:
            c = blue
        elif i in pinky_ids:
            c = purple
        else:
            c = grey
        
        colors[i, :] = c
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path, pcd)

def get_ftip_distances_from_landmarks(landmarks):

    ftip_pos = get_ftip_pos_from_landmarks(landmarks)

    for cur_finger in range(5):
        print(f"Finger {cur_finger} distance to")
        for i in range(5):
            dist = np.linalg.norm(ftip_pos[cur_finger, :] - ftip_pos[i, :])
            print(f"Finger {i}: {dist}")

