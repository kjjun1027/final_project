import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

mp_pose = mp.solutions.pose

def process_frame(frame, pose):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    if result.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        return landmarks
    else:
        return np.zeros((33, 3))

def rotate_landmarks_to_front(landmarks):
    pelvis_center = (landmarks[23] + landmarks[24]) / 2
    return landmarks - pelvis_center


def smooth_landmarks(landmark_list):
    landmark_array = np.array(landmark_list)
    smoothed = []

    for joint in range(33):
        joint_coords = landmark_array[:, joint, :]
        joint_smooth = []
        for axis in range(3):
            y = joint_coords[:, axis]
            x = np.arange(len(y))
            try:
                spline = UnivariateSpline(x, y, s=0.5)
                y_smooth = spline(x)
            except:
                y_smooth = y
            joint_smooth.append(y_smooth)
        joint_smooth = np.stack(joint_smooth, axis=1)
        smoothed.append(joint_smooth)

    smoothed = np.stack(smoothed, axis=1)
    return smoothed

def extract_landmarks(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = process_frame(frame, pose)
        rotated = rotate_landmarks_to_front(landmarks)
        all_landmarks.append(rotated)

    cap.release()
    pose.close()

    if not all_landmarks:
        print("No landmarks found.")
        return

    smoothed = smooth_landmarks(all_landmarks)

    rows = []
    for frame in smoothed:
        row = {}
        for i, (x, y, z) in enumerate(frame):
            row[f"landmark_{i}_x"] = x
            row[f"landmark_{i}_y"] = y
            row[f"landmark_{i}_z"] = z
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    #print(f"Saved CSV to {output_csv}")
