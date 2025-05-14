import numpy as np
import os
import pandas as pd
import pickle
import cv2
import logging
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
from sklearn.preprocessing import normalize
from collections import defaultdict
from src.processing import extract_landmarks
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
import json
import mediapipe as mp
plt.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rc('font', family=font_prop.get_name())

# 1. CSV 데이터 로딩 및 3D 좌표 변환
def load_and_process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.error("❌ CSV 파일이 비어 있습니다.")
            return None
        if not all([f'landmark_{i}_x' in df.columns for i in range(33)]):
            logging.error("❌ CSV 파일에 필수 칼럼이 없습니다.")
            return None
    except Exception as e:
        logging.error(f"❌ CSV 파일 로딩 오류: {str(e)}")
        return None

    logging.info(f"✅ CSV 파일 로딩 성공: {csv_path}")
    logging.info(f"📊 데이터프레임 정보: {df.head()}")
    
    # 기존 처리 로직 그대로 유지
    rename_mapping_x = {f"joint_{i}": f"landmark_{i}_x" for i in range(33)}
    rename_mapping_y = {f"joint_{i+33}": f"landmark_{i}_y" for i in range(33)}
    rename_mapping_z = {f"joint_{i+66}": f"landmark_{i}_z" for i in range(33)}
    df.rename(columns=rename_mapping_x, inplace=True)
    df.rename(columns=rename_mapping_y, inplace=True)
    df.rename(columns=rename_mapping_z, inplace=True)

    coordinate_cols = [col for col in df.columns if '_x' in col or '_y' in col or '_z' in col]
    df[coordinate_cols] = df[coordinate_cols].apply(pd.to_numeric, errors='coerce')

    df.interpolate(method='linear', axis=0, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# 2. 관절 각도 계산
def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    dot_product = np.dot(ba, bc)
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def compute_joint_angles_with_derivatives(df):
    """
    3D 좌표 데이터를 기반으로 10개 관절의 각도, 속도, 가속도를 계산
    """
    joint_angles = []
    for _, row in df.iterrows():
        frame_landmarks = [(row[f'landmark_{i}_x'], row[f'landmark_{i}_y'], row[f'landmark_{i}_z'])
                           for i in range(33)]
        left_knee = compute_angle(frame_landmarks[23], frame_landmarks[25], frame_landmarks[27])
        right_knee = compute_angle(frame_landmarks[24], frame_landmarks[26], frame_landmarks[28])
        left_elbow = compute_angle(frame_landmarks[11], frame_landmarks[13], frame_landmarks[15])
        right_elbow = compute_angle(frame_landmarks[12], frame_landmarks[14], frame_landmarks[16])
        left_shoulder = compute_angle(frame_landmarks[13], frame_landmarks[11], frame_landmarks[23])
        right_shoulder = compute_angle(frame_landmarks[14], frame_landmarks[12], frame_landmarks[24])
        left_hip = compute_angle(frame_landmarks[11], frame_landmarks[23], frame_landmarks[25])
        right_hip = compute_angle(frame_landmarks[12], frame_landmarks[24], frame_landmarks[26])
        left_ankle = compute_angle(frame_landmarks[25], frame_landmarks[27], frame_landmarks[29])
        right_ankle = compute_angle(frame_landmarks[26], frame_landmarks[28], frame_landmarks[30])

        joint_angles.append([left_knee, right_knee, left_elbow, right_elbow,
                             left_shoulder, right_shoulder, left_hip, right_hip,
                             left_ankle, right_ankle])

    joint_angles = np.array(joint_angles)

    #속도 및 가속도 계산 추가
    velocity = np.gradient(joint_angles, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    return joint_angles, velocity, acceleration
    

# 3. 운동 횟수 분석
def detect_repetition_intervals(angle_features, velocity_features, acceleration_features, min_frames=30, 
                                prominence=1, smoothing_window=15):
    """
        각도 변화량 기반으로 0 -> 1 -> 0 반복을 감지하여 운동 횟수를 분석합니다.
    - 각도 변화량의 최대값(1)과 최소값(0) 사이를 반복으로 판단합니다.
    - 노이즈 제거 필터(Savitzky-Golay)를 추가하여 정확도를 개선합니다.
    """
    total_frames = len(angle_features)
    if total_frames < min_frames:
        return [(0, total_frames)]  # 너무 짧으면 전체를 하나의 반복으로 처리

    # 각도, 속도, 가속도의 변화량을 각각 계산
    avg_angles = np.mean(angle_features, axis=1)
    avg_velocity = np.linalg.norm(velocity_features, axis=1)
    avg_acceleration = np.linalg.norm(acceleration_features, axis=1)
    
    # 각도 변화량 계산
    angle_differences = np.abs(np.gradient(avg_angles))
    
    # 노이즈 제거 (Savitzky-Golay 필터)
    if smoothing_window > 1 and smoothing_window < total_frames:
        angle_differences = savgol_filter(angle_differences, smoothing_window, polyorder=3)
    
    # 속도 및 가속도의 평균값 계산
    mean_velocity = np.mean(avg_velocity)
    mean_acceleration = np.mean(avg_acceleration)

    # 기준선 이상인 값만 추출 (속도 및 가속도의 0.4 배 이상인 것만)
    valid_indices = np.where((avg_velocity > 0.4 * mean_velocity) | (avg_acceleration > 0.4 * mean_acceleration))[0]
    angle_differences_filtered = np.zeros_like(angle_differences)
    angle_differences_filtered[valid_indices] = angle_differences[valid_indices]
    
    # 피크와 골짜기 감지
    peaks, _ = find_peaks(angle_differences_filtered, prominence=prominence, distance=min_frames)
    valleys, _ = find_peaks(-angle_differences_filtered, prominence=prominence, distance=min_frames)
    
    # 반복 구간 감지 (0 -> 1 -> 0)
    intervals = []
    i = 0
    while i < len(valleys) - 1:
        start = valleys[i]
        end = valleys[i + 1]
        
        peaks_in_range = [p for p in peaks if start < p < end]
        
        if len(peaks_in_range) > 0:
            intervals.append((start, end))
        
        i += 1

    logging.info(f"🔍 검출된 운동 반복 수: {len(intervals)}")
    return intervals

# 4. 각 반복을 단계별로 분류 (정확도 최적화)
def classify_phases(joint_features, velocity_features, acceleration_features, 
                    cluster_centers, ideal_phase_lengths, ideal_trajectory):
    """
    단계 분류: 이상적 궤적의 단계 배열을 기준으로 매핑.
    """
    min_len = min(len(joint_features), len(ideal_trajectory))
    ideal_phases_resampled = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(ideal_trajectory)),
        ideal_trajectory
    ).round().astype(int)

    num_phases = len(np.unique(ideal_phases_resampled))
    
    computed_progress = [
        np.sum(ideal_phases_resampled == p) / min_len for p in range(num_phases)
    ]

    phase_frames = [{"phase": p, "indices": np.where(ideal_phases_resampled == p)[0]}
                    for p in range(num_phases)]

    return ideal_phases_resampled.tolist(), computed_progress, phase_frames


# 5. 이상적 궤적과 비교 & 유사도 계산 (정확도 최적화, L2 거리 사용)
def compare_phase_similarity(computed_phases, ideal_phases, 
                             computed_velocities, ideal_velocities, 
                             computed_accelerations, ideal_accelerations,
                             computed_progress, ideal_phase_progress, 
                             phase_transitions,
                             actual_trajectory_points, ideal_trajectory_points):
    """
    정확하고 간단한 단계 비교 및 DTW 기반 유사도 계산.
    """
    min_length = min(len(computed_phases), len(ideal_phases))
    computed_phases = computed_phases[:min_length]
    ideal_phases = ideal_phases[:min_length]

    # DTW 유사도
    dtw_distance = dtw.distance(computed_phases, ideal_phases)
    dtw_similarity = 1 / (1 + dtw_distance / min_length)

    # Phase 정확도
    phase_accuracy = np.mean(np.array(computed_phases) == np.array(ideal_phases))

    # 단계 전환 정확도
    computed_transitions = np.where(np.diff(computed_phases) != 0)[0]
    ideal_transitions = np.where(np.diff(ideal_phases) != 0)[0]
    transition_accuracy = 1 / (1 + np.mean([abs(ct - it) for ct, it in zip(computed_transitions, ideal_transitions)])) if computed_transitions.size and ideal_transitions.size else 1

    # 속도 및 가속도 유사도
    velocity_similarity = 1 / (1 + np.mean(np.linalg.norm(computed_velocities - ideal_velocities[ideal_phases], axis=1)))
    acceleration_similarity = 1 / (1 + np.mean(np.linalg.norm(computed_accelerations - ideal_accelerations[ideal_phases], axis=1)))

    # 진행률 정확도
    progress_similarity = 1 - np.mean(np.abs(np.array(computed_progress) - np.array(ideal_phase_progress)))

    # 전환 점수
    transition_score = np.mean([
        1 / (1 + phase_transitions[k]["angle_change"] +
             phase_transitions[k]["velocity_change"] +
             phase_transitions[k]["acceleration_change"])
        for k in phase_transitions
    ]) if phase_transitions else 1

    # 7. 최종 유사도 계산 (가중치 조정)
    total_similarity = (0.35 * dtw_similarity +            # DTW 기반 유사도 (전체 궤적의 패턴 비교)
                        0.2 * phase_accuracy +            # 단계 정확도
                        0.15 * transition_accuracy +      # 단계 전환 정확도
                        0.1 * velocity_similarity +       # 속도 유사도
                        0.1 * acceleration_similarity +   # 가속도 유사도
                        0.1 * progress_similarity)         # 진행률 유사도

    print(f"✅ 유사도: {total_similarity * 100:.2f}% (궤적 유사도: {dtw_similarity * 100:.2f}%, "
          f"단계 정확도: {phase_accuracy * 100:.2f}%, "
          f"전환 정확도: {transition_accuracy * 100:.2f}%, "
          f"속도 유사도: {velocity_similarity * 100:.2f}%, "
          f"가속도 유사도: {acceleration_similarity * 100:.2f}%, "
          f"진행률 유사도: {progress_similarity * 100:.2f}%, "
          f"단계 전환 변화량 유사도: {transition_score} ")

    return total_similarity * 100

# 6. 영상 출력
def extract_first_frame_landmarks(df, segment):
    """
    각 운동 반복 구간의 첫 번째 프레임에서 주요 관절의 좌표를 추출하고 관절 각도를 계산합니다.
    """
    try:
        start_frame = segment[0]
        if start_frame >= len(df):
            logging.error(f"❌ 잘못된 프레임 인덱스: {start_frame}. 데이터 프레임의 길이: {len(df)}")
            return None  # 빈 리스트 대신 None 반환

        # 관절 좌표 추출 (33개 모든 좌표 가져오기)
        frame_landmarks = [(df[f'landmark_{i}_x'].iloc[start_frame], 
                            df[f'landmark_{i}_y'].iloc[start_frame], 
                            df[f'landmark_{i}_z'].iloc[start_frame]) 
                           for i in range(33)]

        if not frame_landmarks:
            logging.error(f"❌ 프레임 {start_frame}에서 관절 좌표를 추출하지 못했습니다.")
            return None

        # 주요 관절 각도를 계산하여 반환
        left_knee = compute_angle(frame_landmarks[23], frame_landmarks[25], frame_landmarks[27])
        right_knee = compute_angle(frame_landmarks[24], frame_landmarks[26], frame_landmarks[28])
        left_elbow = compute_angle(frame_landmarks[11], frame_landmarks[13], frame_landmarks[15])
        right_elbow = compute_angle(frame_landmarks[12], frame_landmarks[14], frame_landmarks[16])
        left_shoulder = compute_angle(frame_landmarks[13], frame_landmarks[11], frame_landmarks[23])
        right_shoulder = compute_angle(frame_landmarks[14], frame_landmarks[12], frame_landmarks[24])
        left_hip = compute_angle(frame_landmarks[11], frame_landmarks[23], frame_landmarks[25])
        right_hip = compute_angle(frame_landmarks[12], frame_landmarks[24], frame_landmarks[26])
        left_ankle = compute_angle(frame_landmarks[25], frame_landmarks[27], frame_landmarks[29])
        right_ankle = compute_angle(frame_landmarks[26], frame_landmarks[28], frame_landmarks[30])

        key_angles = [
            left_knee, right_knee, left_elbow, right_elbow,
            left_shoulder, right_shoulder, left_hip, right_hip,
            left_ankle, right_ankle
        ]
        
        logging.info(f"✅ 주요 관절 각도 추출 성공 (프레임: {start_frame})")
        return key_angles
    except Exception as e:
        logging.error(f"❌ 주요 관절 좌표 추출 오류: {str(e)}")
        return None


def test_generate_ideal_trajectory(first_frame_angles, ideal_angles, ideal_velocities, ideal_accelerations, ideal_progress):
    """
    초기 각도를 기반으로 이상적 궤적을 생성합니다.
    """
    if first_frame_angles is None:
        logging.error("❌ 초기 각도 데이터가 없습니다.")
        return None

    ideal_trajectory = [first_frame_angles]

    for t, (angle_set, velocity_set, acceleration_set) in enumerate(zip(ideal_angles, ideal_velocities, ideal_accelerations)):
        new_frame_angles = []

        progress_factor = ideal_progress[t] if t < len(ideal_progress) else 1.0

        for j, base_angle in enumerate(first_frame_angles):
            angle_change = angle_set[j % len(angle_set)] * progress_factor
            velocity_change = velocity_set[j % len(velocity_set)] * progress_factor
            acceleration_change = acceleration_set[j % len(acceleration_set)] * progress_factor

            new_angle = base_angle + angle_change + velocity_change * 0.01 + acceleration_change * 0.001
            new_frame_angles.append(new_angle)
        
        ideal_trajectory.append(new_frame_angles)

    return ideal_trajectory

def plot_trajectory_comparison(
    actual_trajectory,            # numpy array (프레임 x 관절)
    ideal_trajectory,             # numpy array (프레임 x 관절)
    save_path,                    # 저장 경로
    exercise_name="Exercise"      # 운동 이름
):

    actual_trajectory = np.array(actual_trajectory)
    ideal_trajectory = np.array(ideal_trajectory)

    # 1. 각도, 속도, 가속도 계산
    actual_velocity = np.gradient(actual_trajectory, axis=0)
    ideal_velocity = np.gradient(ideal_trajectory, axis=0)

    actual_acceleration = np.gradient(actual_velocity, axis=0)
    ideal_acceleration = np.gradient(ideal_velocity, axis=0)

    # 2. 단계 임의 추정
    phase_actual = estimate_phases(actual_trajectory)
    phase_ideal = estimate_phases(ideal_trajectory)

    # 3. 단계 전환 변화량
    transition_actual = np.abs(np.diff(actual_trajectory, axis=0))
    transition_ideal = np.abs(np.diff(ideal_trajectory, axis=0))
    avg_actual_transition = np.mean(transition_actual, axis=1)
    avg_ideal_transition = np.mean(transition_ideal, axis=1)

    # 4. 각도 변화량 (Angle Change Rate)
    angle_change_actual = np.linalg.norm(np.diff(actual_trajectory, axis=0), axis=1)
    angle_change_ideal = np.linalg.norm(np.diff(ideal_trajectory, axis=0), axis=1)

    x = np.arange(len(actual_trajectory))

    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)

    # 1. Phase 비교
    axes[0].plot(x, phase_actual, label="Actual Phase", color='blue')
    axes[0].plot(x, phase_ideal, label="Ideal Phase", color='red', linestyle='--')
    axes[0].set_title("Phase Comparison")
    axes[0].legend()
    axes[0].grid(True)

    # 2. 속도 평균
    axes[1].plot(x, np.mean(actual_velocity, axis=1), label="Actual", color='blue')
    axes[1].plot(x, np.mean(ideal_velocity, axis=1), label="Ideal", color='red', linestyle='--')
    axes[1].set_title("Velocity Comparison")
    axes[1].legend()
    axes[1].grid(True)

    # 3. 가속도 평균
    axes[2].plot(x, np.mean(actual_acceleration, axis=1), label="Actual", color='blue')
    axes[2].plot(x, np.mean(ideal_acceleration, axis=1), label="Ideal", color='red', linestyle='--')
    axes[2].set_title("Acceleration Comparison")
    axes[2].legend()
    axes[2].grid(True)

    # 4. 단계 전환 변화량
    axes[3].plot(x[:-1], avg_actual_transition, label="Actual", color='blue')
    axes[3].plot(x[:-1], avg_ideal_transition, label="Ideal", color='red', linestyle='--')
    axes[3].set_title("Phase Transition Δ (Mean of Joint Differences)")
    axes[3].legend()
    axes[3].grid(True)

    # ✅ 5. 각도 변화량
    axes[4].plot(x[1:], angle_change_actual, label="Actual", color='blue')
    axes[4].plot(x[1:], angle_change_ideal, label="Ideal", color='red', linestyle='--')
    axes[4].set_title("Angle Change Rate (L2 Norm per Frame)")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 시각화 그래프 저장 완료: {save_path}")

def estimate_phases(joint_angles):
    """
    joint_angles (N x J) 데이터를 기반으로 임시 단계 구간을 단순 생성 (예시용).
    나중에 수정 가능.
    """
    # 단순히 각도 평균의 변화량으로 단계 구분 (예시)
    diff = np.abs(np.gradient(np.mean(joint_angles, axis=1)))
    threshold = np.percentile(diff, 60)
    phases = (diff > threshold).astype(int)
    return np.cumsum(phases) % 5  # 최대 5단계로 순환




# 7. 뉴네오슈퍼 애니메이션
def save_ideal_angles_to_csv(ideal_trajectory, save_path="ideal_angles.csv"):
    """
    generate_ideal_trajectory()로 생성한 이상 궤적 (관절 각도 시퀀스)을 CSV로 저장
    - 각 프레임마다 10개 각도를 저장
    """
    if not ideal_trajectory or not isinstance(ideal_trajectory, list):
        print("❌ 잘못된 입력입니다.")
        return
    
    df = pd.DataFrame(ideal_trajectory, columns=[
        f"angle_{i}" for i in range(len(ideal_trajectory[0]))
    ])
    df.to_csv(save_path, index=False)
    print(f"✅ 이상 궤적 CSV 저장 완료: {save_path}")

def save_subset_landmarks_csv(input_csv_path, output_csv_path):
    """
    기존 좌표 CSV에서 주요 관절(14개)의 x, y, z 컬럼만 추출하여 새로운 CSV로 저장
    """
    df = pd.read_csv(input_csv_path)

    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
    columns_to_keep = []

    for i in key_joints:
        columns_to_keep.extend([
            f"landmark_{i}_x",
            f"landmark_{i}_y",
            f"landmark_{i}_z"
        ])

    subset_df = df[columns_to_keep]
    subset_df.to_csv(output_csv_path, index=False)
    print(f"✅ 주요 관절만 추출하여 저장 완료: {output_csv_path}")

def draw_pose_from_points(img, points, offset_x=0, color=(0, 255, 0)):
    """
    10개 포인트(각도 기반)를 기준으로 사람 형태의 연결선 그리기
    인덱스 순서 기준: [LK, RK, LE, RE, LS, RS, LH, RH, LA, RA]
    """

    connections = [
        (4, 6), (6, 0), (0, 8),  # 왼쪽 어깨 → 엉덩이 → 무릎 → 발목
        (5, 7), (7, 1), (1, 9),  # 오른쪽 어깨 → 엉덩이 → 무릎 → 발목
        (4, 5),                 # 어깨 연결
        (4, 2), (5, 3),         # 어깨 → 팔꿈치
    ]

    for x, y in points:
        x = np.clip(x + offset_x, 0, img.shape[1] - 1)
        y = np.clip(y, 0, img.shape[0] - 1)
        cv2.circle(img, (x, y), 5, color, -1)

    for start, end in connections:
        try:
            x1, y1 = points[start]
            x2, y2 = points[end]
            cv2.line(img, (x1 + offset_x, y1), (x2 + offset_x, y2), color, 2)
        except:
            continue



def angle_to_point(angle):
    rad = np.radians(angle)
    x = np.cos(rad)
    y = np.sin(rad)
    z = angle / 180.0
    return x, y, z

def project_point(x, y, z, scale=300, cx=360, cy=520):
    z = -z
    factor = 2.0 / (2.0 - z + 1e-6)
    x2d = int(x * factor * scale + cx)
    y2d = int(y * factor * scale + cy)
    return x2d, y2d

def animate_full_pose_with_ideal_overlay(actual_df, ideal_df, save_path, preview=False):
    import cv2
    import numpy as np

    ACTUAL_POSE_CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 11), (24, 12), (23, 24),
        (23, 25), (25, 27), (27, 29),
        (24, 26), (26, 28), (28, 30)
    ]

    angle_groups = {
        "left_arm": [13, 15],
        "right_arm": [14, 16],
        "left_leg": [25, 27, 29],
        "right_leg": [26, 28, 30]
    }

    # 새로 추가: 관절쌍 벡터 정보
    pair_indices = [
        (13, 15), (14, 16),
        (11, 13), (12, 14),
        (23, 25), (24, 26),
        (25, 27), (26, 28),
        (11, 23), (12, 24),
        (27, 29), (28, 30),
    ]

    h, w = 1040, 720
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    num_frames = min(len(actual_df), len(ideal_df))

    key_joints = [11,12,13,14,15,16,23,24,25,26,27,28,29,30]

    for i in range(num_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)

        actual_points = {}
        for j in key_joints:
            try:
                x = float(actual_df.iloc[i][f"landmark_{j}_x"])
                y = float(actual_df.iloc[i][f"landmark_{j}_y"])
                x2d = int((x * 300) + 360)
                y2d = int((y * 300) + 520)
                actual_points[j] = (x2d, y2d)
            except:
                actual_points[j] = (0, 0)

        # 🧠 이상적 좌표 생성 (관절쌍 벡터 + 각도 기반)
        angle_values = list(ideal_df.iloc[i])
        angle_idx = 0
        ideal_points = actual_points.copy()

        for (j1, j2) in pair_indices:
            if angle_idx >= len(angle_values):
                break
            try:
                angle = float(angle_values[angle_idx])
                angle_idx += 1

                # 실제 벡터 방향 계산
                x1, y1 = actual_df.iloc[i][f"landmark_{j1}_x"], actual_df.iloc[i][f"landmark_{j1}_y"]
                x2, y2 = actual_df.iloc[i][f"landmark_{j2}_x"], actual_df.iloc[i][f"landmark_{j2}_y"]
                dx, dy = x2 - x1, y2 - y1
                norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
                dx, dy = dx / norm, dy / norm

                # 관절 방향 벡터로 이상 좌표 생성
                magnitude = angle / 180.0 * 0.1
                base_x, base_y = float(actual_df.iloc[i][f"landmark_{j1}_x"]), float(actual_df.iloc[i][f"landmark_{j1}_y"])
                ideal_x = base_x + dx * magnitude
                ideal_y = base_y + dy * magnitude

                # 시각화용 2D 투영
                x2d = int((ideal_x * 300) + 360 - 20)  # 오른쪽으로 살짝 보이게
                y2d = int((ideal_y * 300) + 520)
                ideal_points[j2] = (x2d, y2d)

            except Exception as e:
                continue

        # 실제 스켈레톤 그리기 (초록색)
        for start, end in ACTUAL_POSE_CONNECTIONS:
            pt1 = actual_points.get(start, (0, 0))
            pt2 = actual_points.get(end, (0, 0))
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # 이상적 연결선 그리기 (빨간색)
        for group in angle_groups.values():
            for j in range(len(group) - 1):
                a, b = group[j], group[j+1]
                pt1 = ideal_points.get(a, (0, 0))
                pt2 = ideal_points.get(b, (0, 0))
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

        # 점 그리기
        for j, pt in actual_points.items():
            if pt != (0, 0):
                cv2.circle(img, pt, 3, (0, 255, 0), -1)
        for j in ideal_points:
            pt = ideal_points[j]
            if pt != (0, 0):
                cv2.circle(img, pt, 5, (0, 0, 255), -1)

        cv2.putText(img, "Red: Ideal (from joint pairs) | Green: Actual", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        out.write(img)
        if preview:
            cv2.imshow("Pose Overlay", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    if preview:
        cv2.destroyAllWindows()

    return save_path





def project_relative_to_actual(x, y, z, actual_origin, is_leg=False):
    ax, ay = actual_origin
    # 다리는 더 작은 스케일로 덜 튀게 보정
    scale_xy = 30 if is_leg else 45
    scale_z = 10 if is_leg else 18
    x2d = int(ax + x * scale_xy)
    y2d = int(ay + y * scale_xy + z * scale_z)
    return x2d, y2d


def extract_joint_angle_sequence(df, start, end):
    angle_seq = []
    for i in range(start, end):
        try:
            frame = [(df[f'landmark_{j}_x'].iloc[i], 
                      df[f'landmark_{j}_y'].iloc[i], 
                      df[f'landmark_{j}_z'].iloc[i]) for j in range(33)]
            angles = [
                compute_angle(frame[23], frame[25], frame[27]),
                compute_angle(frame[24], frame[26], frame[28]),
                compute_angle(frame[11], frame[13], frame[15]),
                compute_angle(frame[12], frame[14], frame[16]),
                compute_angle(frame[13], frame[11], frame[23]),
                compute_angle(frame[14], frame[12], frame[24]),
                compute_angle(frame[11], frame[23], frame[25]),
                compute_angle(frame[12], frame[24], frame[26]),
                compute_angle(frame[25], frame[27], frame[29]),
                compute_angle(frame[26], frame[28], frame[30]),
            ]
            angle_seq.append(angles)
        except Exception as e:
            print(f"❌ 관절 각도 추출 실패 (프레임 {i}): {e}")
            continue
    return angle_seq

def interpolate_trajectory_by_phase_progress(trajectory, phase_progress, target_length):
    """
    관절 각도 궤적을 단계별 진행률 기준으로 target_length 길이로 보간.
    
    trajectory: 프레임별 phase 인덱스 배열
    phase_progress: 각 단계의 비율
    target_length: 목표 프레임 수
    """
    original_length = len(trajectory)
    if original_length < 2 or target_length < 2:
        return trajectory

    # 각 단계의 시작과 끝 인덱스를 원본/목표 프레임 기준으로 계산
    original_phase_boundaries = np.cumsum([0] + [int(p * original_length) for p in phase_progress])
    target_phase_boundaries = np.cumsum([0] + [int(p * target_length) for p in phase_progress])

    # 끝점 보정
    original_phase_boundaries[-1] = original_length
    target_phase_boundaries[-1] = target_length

    interpolated_traj = []

    for i in range(len(phase_progress)):
        start_o, end_o = original_phase_boundaries[i], original_phase_boundaries[i+1]
        start_t, end_t = target_phase_boundaries[i], target_phase_boundaries[i+1]

        segment = trajectory[start_o:end_o]

        if len(segment) < 2:
            segment = [segment[0]] * 2  # 최소한 2개의 값으로 복사

        x_old = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, end_t - start_t)

        # 단계의 각도를 보간
        interpolator = interp1d(x_old, segment, kind='linear', fill_value="extrapolate")
        interpolated_segment = interpolator(x_new)

        interpolated_traj.extend(interpolated_segment.tolist())

    # 보정하여 최종 길이 맞추기
    while len(interpolated_traj) < target_length:
        interpolated_traj.append(interpolated_traj[-1])

    return interpolated_traj[:target_length]

# 8. 프롬프트 생성

# === 위치 기반 오차 분석 (실제 방향 기반) ===
def compute_projected_ideal_points_realistic(actual_df, ideal_df):
    key_joints = [11,12,13,14,15,16,23,24,25,26,27,28,29,30]
    pair_indices = [
        (13, 15), (14, 16),  # 팔꿈치
        (11, 13), (12, 14),  # 어깨
        (23, 25), (24, 26),  # 엉덩이
        (25, 27), (26, 28),  # 무릎
        (11, 23), (12, 24),     # 어깨 → 힙 (Hip 좌표 생성용)
        (27, 29), (28, 30),
    ]

    ideal_points_per_frame = []

    for i in range(len(ideal_df)):
        actual_points = {}
        for j in key_joints:
            try:
                x = actual_df.loc[i, f"landmark_{j}_x"]
                y = actual_df.loc[i, f"landmark_{j}_y"]
                z = actual_df.loc[i, f"landmark_{j}_z"]
                actual_points[j] = np.array([x, y, z])
            except:
                actual_points[j] = np.array([0, 0, 0])

        angles = ideal_df.iloc[i].values
        ideal_frame_points = {}

        for idx, (j1, j2) in enumerate(pair_indices):
            if idx >= len(angles): continue
            base = actual_points[j1]
            direction = actual_points[j2] - actual_points[j1]
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.array([1, 0, 0])
            else:
                direction /= norm
            magnitude = angles[idx] / 180.0 * 0.1
            new_point = base + direction * magnitude
            ideal_frame_points[j2] = tuple(new_point)

        for j in key_joints:
            if j not in ideal_frame_points:
                ideal_frame_points[j] = tuple(actual_points[j])

        ideal_points_per_frame.append([ideal_frame_points[j] for j in key_joints])

    return ideal_points_per_frame

def summarize_joint_errors_for_feedback(
        error_df, 
        joint_labels, 
        actual_points_per_frame, 
        ideal_points_per_frame, 
        SessionID:str,
        UserId:str,
        ExercisesName:str,
        frame_group_thresh=3
    ):

    joint_thresholds = {
        joint: np.mean(error_df[joint]) + 1.5 * np.std(error_df[joint])
        for joint in joint_labels
    }

    frame_flags = defaultdict(list)
    for _, row in error_df.iterrows():
        frame = int(row["frame"])
        for joint in joint_labels:
            if row[joint] > joint_thresholds[joint]:
                frame_flags[frame].append((joint, row[joint]))

    def group_frames(frames):
        grouped, temp, last = [], [], None
        for f in sorted(frames):
            if last is None or f == last + 1:
                temp.append(f)
            else:
                grouped.append(temp)
                temp = [f]
            last = f
        if temp:
            grouped.append(temp)
        return grouped

    def get_direction_vector(diff_vec):
        directions = []
        if abs(diff_vec[0]) > 0.01:
            directions.append("to the right" if diff_vec[0] > 0 else "to the left")
        if abs(diff_vec[1]) > 0.01:
            directions.append("upward" if diff_vec[1] > 0 else "downward")
        if abs(diff_vec[2]) > 0.01:
            directions.append("forward" if diff_vec[2] > 0 else "backward")
        return ", ".join(directions) if directions else "minor deviation"

    def format_error_detail(avg_error, direction_str):
        if direction_str == "minor deviation":
            return "No significant deviation detected."
        if avg_error < 0.02:
            level = "slightly"
        elif avg_error < 0.05:
            level = "a bit"
        elif avg_error < 0.1:
            level = "moderately"
        elif avg_error < 0.2:
            level = "significantly"
        else:
            level = "severely"
        return f"{level} deviated {direction_str}"

    grouped_frames = group_frames(frame_flags.keys())
    result = []
    result.append({
        "SessionID": f"{SessionID}",
        "UserID": f"{UserId}",
        "ExerciseName": f"{ExercisesName}",
    })
    for group in grouped_frames:
        if len(group) < frame_group_thresh:
            continue

        joint_stats = defaultdict(list)
        joint_directions = defaultdict(list)
        for f in group:
            for joint, err in frame_flags[f]:
                joint_stats[joint].append(err)
                j_index = joint_labels.index(joint)
                actual = np.array(actual_points_per_frame[f][j_index])
                ideal = np.array(ideal_points_per_frame[f][j_index])
                diff = actual - ideal
                joint_directions[joint].append(diff)

        summary = []
        for joint, errors in joint_stats.items():
            if len(errors) >= frame_group_thresh:
                avg_error = round(np.mean(errors), 4)
                diffs = np.array(joint_directions[joint])
                avg_vec = np.mean(diffs, axis=0)
                direction_str = get_direction_vector(avg_vec)
                error_detail = format_error_detail(avg_error, direction_str)

                summary.append({
                    "name": joint,
                    "error_detail": error_detail
                })

        if summary:
            result.append({
                "range": f"Frame {group[0]}~{group[-1]}",
                "joints": summary,
                "comment": "Multiple joint deviations detected during this interval."
            })

    return result

def get_user_center_from_frame(frame: np.ndarray) -> tuple:
    """
    Extracts the user's pelvis center from a frame using MediaPipe Pose.
    Returns (x, y) normalized coordinates or None if detection fails.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        if len(landmarks) < 25:
            return None

        # landmarks 23: left_hip, 24: right_hip
        center_x = (landmarks[23].x + landmarks[24].x) / 2
        center_y = (landmarks[23].y + landmarks[24].y) / 2

        return (center_x, center_y)

def generate_feedback_overlay_images(
    video_path: str,
    feedback_json_path: str,
    actual_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    output_dir: str
):
    os.makedirs(os.path.join(output_dir, "frame_feedback"), exist_ok=True)
    key_joints = [11,12,13,14,15,16,23,24,25,26,27,28,29,30]
    connections = [(11,13),(13,15),(12,14),(14,16),(23,25),
                   (25,27),(27,29),(24,26),(26,28),(28,30)]

    with open(feedback_json_path, "r", encoding="utf-8") as f:
        feedback_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for item in feedback_data:
        if "range" not in item:
            continue

        start, end = map(int, item["range"].replace("Frame ", "").split("~"))
        if (end - start + 1) < 20:
            continue

        target_frame = (start + end) // 2
        if target_frame >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        if not success:
            print(f"❌ Frame {target_frame} capture failed")
            continue
        h, w = frame.shape[:2]

        # ✅ 사용자 중심 좌표 구하기
        user_center = get_user_center_from_frame(frame)
        if user_center is None:
            print(f"❌ Failed to detect user center in Frame {target_frame}")
            continue

        user_center_x, user_center_y = user_center
        offset_x = int((1.0 - user_center_x) * w)
        offset_y = h // 2 - int(user_center_y * h)
        #print(f"{offset_x},{offset_y}")

        def to_pixel_coords(x, y):
            return (int(x * w + offset_x), int(y * h + offset_y))

        actual_points = []
        ideal_points = []

        for i, j in enumerate(key_joints):
            try:
                ax = actual_df.loc[target_frame, f"landmark_{j}_x"]
                ay = actual_df.loc[target_frame, f"landmark_{j}_y"]
                actual_points.append((ax, ay))
            except:
                actual_points.append((0, 0))

            try:
                angle = ideal_df.iloc[target_frame, i]
                dx = 0.05 if j % 2 == 0 else -0.05
                dy = -0.05
                ix = ax + dx * (angle / 180)
                iy = ay + dy * (angle / 180)
                ideal_points.append((ix, iy))
            except:
                ideal_points.append((0, 0))

        actual_points_pixel = [to_pixel_coords(x, y) for (x, y) in actual_points]
        ideal_points_pixel = [to_pixel_coords(x, y) for (x, y) in ideal_points]

        overlay = frame.copy()

        # draw actual (green)
        for (i, j) in connections:
            i1 = key_joints.index(i)
            i2 = key_joints.index(j)
            if actual_points[i1] != (0,0) and actual_points[i2] != (0,0):
                cv2.line(overlay, actual_points_pixel[i1], actual_points_pixel[i2], (0,255,0), 2)
        for pt in actual_points_pixel:
            if pt != (0,0): cv2.circle(overlay, pt, 4, (0,255,0), -1)

        # draw ideal (red)
        for (i, j) in connections:
            i1 = key_joints.index(i)
            i2 = key_joints.index(j)
            if ideal_points[i1] != (0,0) and ideal_points[i2] != (0,0):
                cv2.line(overlay, ideal_points_pixel[i1], ideal_points_pixel[i2], (0,0,255), 2)
        for pt in ideal_points_pixel:
            if pt != (0,0): cv2.circle(overlay, pt, 4, (0,0,255), -1)

        cv2.putText(overlay, f"Frame {target_frame}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        save_path = os.path.join(output_dir)
        cv2.imwrite(save_path, overlay)
        #print(f"✅ Saved: {save_path}")

    cap.release()


def build_gpt_prompt_from_summary(
    feedback_data: list
) -> str:
    # 1. Parse metadata (assumed to be the first dictionary with "ExerciseName")
    meta = None
    for entry in feedback_data:
        if "ExerciseName" in entry:
            meta = entry
            break

    if meta is None:
        raise ValueError("No metadata with 'ExerciseName' found in feedback_data.")

    exercise_name = meta.get("ExerciseName", "Unknown Exercise")
    exercise_description = meta.get("ExerciseDescription", "No description provided.")
    ideal_pose_bullet_points = meta.get("IdealPose", [])

    # 2. Construct prompt string with manual line breaks
    prompt = ""
    prompt += f"🧠 You are a movement feedback coach. The user is performing '{exercise_name}'.\n\n"
    prompt += f"Exercise Description: {exercise_description}\n\n"
    prompt += "Ideal Form:\n"
    for bullet in ideal_pose_bullet_points:
        prompt += f"• {bullet}\n"

    prompt += "\nDetected Deviations:\n"
    
    # 3. Add deviation summaries (skip any metadata-like entries)
    for section in feedback_data:
        if "range" not in section or "joints" not in section:
            continue
        prompt += f"\n[📍 {section['range']}]\n"
        for joint in section["joints"]:
            prompt += f"- {joint['name']}: {joint['error_detail']}\n"

    # 4. GPT instruction
    prompt += (
        "\n\nBased on the above, please explain in three parts:\n"
        "1. What is incorrect with the user's form\n"
        "2. Why this might be happening\n"
        "3. How to correct it (include light stretches or drills if helpful)\n\n"
        "Avoid technical jargon. Make your explanation simple and actionable for beginners.\n"
    )

    return prompt


# 9. 메인 실행
def main():

    if len(sys.argv) != 5:
        print("❗사용법: python analysis.py [영상경로] [출력이미지경로] [운동이름] [임시파일일]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    exercise_name = sys.argv[3]
    temp_path = sys.argv[4]
    sub_exercise = f"{exercise_name}v2"

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"{temp_path}/{video_filename}.csv"
    os.makedirs(temp_path, exist_ok=True)

    extracted_csv = extract_landmarks(video_path, csv_path)
    print(f"🔍 extract_landmarks 결과: {extracted_csv}")
    print(f"🔍 CSV Generated: {csv_path}, Proceeding to Similarity Analysis...")

    df = load_and_process_csv(csv_path)
    if df is None or df.empty:
        logging.error("❌ CSV 파일 로딩 실패 또는 데이터가 비어 있습니다.")
        return

    angle_features, velocity_features, acceleration_features = compute_joint_angles_with_derivatives(df)
    if angle_features.size == 0:
        logging.error("❌ 관절 각도 데이터 추출 실패.")
        return

    segments = detect_repetition_intervals(angle_features, velocity_features, acceleration_features)
    if not segments:
        logging.error("❌ 운동 반복 구간 검출 실패.")
        return
    print(f"🔍 검출된 운동 반복 수: {len(segments)}회")

    # StepSplit 기준 데이터 로드
    split_criteria_path = f"./data/models/StepSplit/{sub_exercise}_combined_step_split_criteria.pkl"
    if not os.path.exists(split_criteria_path):
        logging.error("❌ 기준 데이터 파일을 찾을 수 없습니다.")
        return

    with open(split_criteria_path, "rb") as f:
        split_criteria = pickle.load(f)

    cluster_centers = np.hstack([
        split_criteria["angles"],
        split_criteria["velocities"],
        split_criteria["accelerations"]
    ])
    ideal_phase_lengths = split_criteria["phase_lengths"]
    ideal_phase_progress = split_criteria["phase_progress"]

    # 수정된 이상적 궤적 로드 (관절별로 저장)
    ideal_trajectory_path = f"./data/models/analysistrajectory/{sub_exercise}/{sub_exercise}_ideal_trajectory.pkl"
    if not os.path.exists(ideal_trajectory_path):
        logging.error("❌ 이상적 궤적 파일을 찾을 수 없습니다.")
        return

    with open(ideal_trajectory_path, "rb") as f:
        ideal_joint_data = pickle.load(f)  # 관절별 데이터 로드

    joint_names = [f"joint_{i}" for i in range(10)]
    actual_all_angles = []
    ideal_all_angles = []
    total_similarity = []

    for i, (start, end) in enumerate(segments):
        actual_segment_angles = angle_features[start:end]
        actual_segment_velocities = velocity_features[start:end]
        actual_segment_accelerations = acceleration_features[start:end]

        actual_all_angles.extend(actual_segment_angles.tolist())

        # 관절별로 보간된 이상적 궤적 생성
        generated_ideal_points = []
        for joint in joint_names:
            joint_data = ideal_joint_data[joint]
            ideal_trajectory_phases = joint_data["ideal_trajectory"]
            ideal_phase_progress_segment = joint_data["phase_progress"]

            # 보간된 이상적 각도 생성 (기준 데이터를 단계별로 보간)
            interpolated_ideal_angles = interpolate_trajectory_by_phase_progress(
                ideal_trajectory_phases,
                ideal_phase_progress_segment,
                target_length=(end - start)
            )
            generated_ideal_points.append(interpolated_ideal_angles)

        generated_ideal_points = np.array(generated_ideal_points).T
        ideal_all_angles.extend(generated_ideal_points.tolist())

        # 단계 계산 및 유사도 분석
        try:
            computed_phases, computed_progress, _ = classify_phases(
                actual_segment_angles, 
                actual_segment_velocities, 
                actual_segment_accelerations, 
                cluster_centers, 
                ideal_phase_lengths,
                ideal_trajectory_phases
            )

            similarity = compare_phase_similarity(
                computed_phases,
                ideal_trajectory_phases,
                actual_segment_velocities,
                split_criteria["velocities"],
                actual_segment_accelerations,
                split_criteria["accelerations"],
                computed_progress,
                ideal_phase_progress_segment,
                joint_data["phase_transitions"],  # 단계 전환 정보 사용
                actual_segment_angles,
                generated_ideal_points
            )
            total_similarity.append(similarity)

        except Exception as e:
            logging.error(f"❌ 유사도 계산 오류: {str(e)}")
            total_similarity.append(0)

        output_path = os.path.join(temp_path, f"babellow_trajectory_{i+1}.png")
        plot_trajectory_comparison(
            actual_segment_angles,
            generated_ideal_points,
            save_path=output_path,
            exercise_name="Babellow"
        )

    try:
        overall_similarity = np.average(
            total_similarity, 
            weights=[(end - start) for start, end in segments]
        )
    except Exception as e:
        logging.error(f"❌ 전체 평균 유사도 계산 오류: {str(e)}")
        overall_similarity = np.mean(total_similarity)
    print(f"\n🔥 전체 평균 유사도: {overall_similarity:.2f}%")

    ideal = os.path.join(temp_path, "ideal_angles.csv")
    actual = os.path.join(temp_path, "actual_angles.csv")

    # 결과 CSV 저장
    if actual_all_angles and ideal_all_angles:
        save_ideal_angles_to_csv(ideal_all_angles, ideal)
        save_ideal_angles_to_csv(actual_all_angles, actual)

        # 전체 프레임 landmark CSV 로드
        actual_landmark_df = pd.read_csv(csv_path)

        # 시각화 영상 생성
        save_video_path = os.path.join(temp_path, "final_pose_comparison.mp4")
        animate_full_pose_with_ideal_overlay(
            actual_df=actual_landmark_df,
            ideal_df=pd.read_csv(ideal),
            save_path=save_video_path,
            preview=True
        )

        print("\n📏 주요 관절별 위치 비교 시작...")
        key_joints = [11,12,13,14,15,16,23,24,25,26,27,28,29,30]
        actual_points_per_frame = []
        for i in range(len(actual_landmark_df)):
            points = []
            for j in key_joints:
                x = actual_landmark_df.loc[i, f"landmark_{j}_x"]
                y = actual_landmark_df.loc[i, f"landmark_{j}_y"]
                z = actual_landmark_df.loc[i, f"landmark_{j}_z"]
                points.append((x, y, z))
            actual_points_per_frame.append(points)

        ideal_angles_df = pd.read_csv(ideal)
        ideal_points_per_frame = compute_projected_ideal_points_realistic(actual_landmark_df, ideal_angles_df)

        joint_errors = {j: [] for j in range(10)}
        for f in range(len(ideal_points_per_frame)):
            for j in range(10):
                ax, ay, az = actual_points_per_frame[f][j]
                ix, iy, iz = ideal_points_per_frame[f][j]
                dist = np.sqrt((ax - ix)**2 + (ay - iy)**2 + (az - iz)**2)
                joint_errors[j].append(dist)

        print("📊 관절별 평균 위치 오차 (단위: 거리)")
        joint_labels = [
            "Left Knee", "Right Knee", "Left Elbow", "Right Elbow",
            "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip",
            "Left Ankle", "Right Ankle"
        ]
        for j in range(10):
            avg_error = np.mean(joint_errors[j])
            print(f"{joint_labels[j]:<15}: {avg_error:.4f}")
            
        # === 📄 프레임별 오차 CSV 저장 ===
        error_csv_path = os.path.join(temp_path, "joint_errors_by_frame.csv")
        frame_data = []
        for f in range(len(joint_errors[0])):
            row = {"frame": f}
            for j in range(10):
                row[joint_labels[j]] = joint_errors[j][f]
            frame_data.append(row)
        error_df = pd.DataFrame(frame_data)
        error_df.to_csv(error_csv_path, index=False)
        print(f"\n📄 프레임별 관절 오차 CSV 저장 완료: {error_csv_path}")

        # === ⚠️ 오차 큰 프레임 탐지 ===
        print("\n⚠️ 오차가 큰 프레임 및 관절 탐지 중...")

        # 관절별 평균 + 표준편차 기반 threshold 계산
        joint_thresholds = {}
        for j in range(10):
            joint_mean = np.mean(joint_errors[j])
            joint_std = np.std(joint_errors[j])
            joint_thresholds[joint_labels[j]] = joint_mean + 1.5 * joint_std

        flagged = []

        for idx, row in error_df.iterrows():
            flagged_joints = []
            for j, joint in enumerate(joint_labels):
                error = row[joint]
                if error > joint_thresholds[joint]:
                    flagged_joints.append((joint, error))
            if len(flagged_joints) >= 3:  # 동시에 3개 이상 틀렸을 때만 표시
                print(f"- Frame {int(row['frame'])}:")
                for joint, err in flagged_joints:
                    print(f"   • {joint}: {err:.4f}")
                flagged.append(idx)

        print(f"\n총 {len(flagged)}개 프레임에서 오차가 큰 관절 다수 발견됨.")
        session_id = "c51f71e1-749a-4f9b-9f55-xxxxxx"
        user_id = "test_user_001"
        exercise_name = "ballow"
        feedback_summary = summarize_joint_errors_for_feedback(
            error_df, joint_labels,
            actual_points_per_frame,    # ← 추가
            ideal_points_per_frame,      # ← 추가
            session_id,
            user_id,
            exercise_name
        )
        json_path = os.path.join(temp_path, "feedback_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(feedback_summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(feedback_summary))
        #print(f"\n✅ 피드백 요약 JSON 저장 완료: {json_path}")

        feedback_data = [{
            "SessionID": session_id,
            "UserID": user_id,
            "ExerciseName": exercise_name,
            "ExerciseDescription": "This exercise strengthens the biceps...",
            "IdealPose": [
                "Keep your elbows fixed at your sides throughout the movement.",
                "Avoid moving your shoulders; they should remain stable.",
                "Maintain a neutral wrist position without bending.",
                "The elbow flexion angle should range between approximately 50 and 140 degrees."
            ]
        }] + feedback_summary

        # feedback_summary는 summarize_joint_errors_for_feedback() 결과 그대로 전달
        prompt = build_gpt_prompt_from_summary(feedback_data)

        # 파일로 저장해도 OK
        gpt_prompt_path = os.path.join(temp_path, "gpt_prompt.json")
        with open(gpt_prompt_path, "w", encoding="utf-8") as f:
            json.dump({ "prompt": prompt }, f, ensure_ascii=False, indent=2)

        with open(os.path.join(temp_path, "gpt_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        #print(f"✅ GPT 프롬프트 JSON 저장 완료: {gpt_prompt_path}")
        generate_feedback_overlay_images(
            video_path=video_path,
            feedback_json_path=f"{temp_path}/feedback_summary.json",
            actual_df=actual_landmark_df,
            ideal_df=ideal_angles_df,
            output_dir=output_dir
        )
        
if __name__ == "__main__":
    main()
