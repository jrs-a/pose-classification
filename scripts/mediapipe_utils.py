import numpy as np

def convert_mediapipe_to_openpose(mediapipe_landmarks_list):
    """
    Converts MediaPipe pose landmarks to OpenPose format.
    Args:
        mediapipe_landmarks_list: A list of lists, where each inner list contains NormalizedLandmark objects for a person.
    Returns:
        A list of dictionaries, each representing keypoints for a person in OpenPose format (25 keypoints: [x, y, confidence]).
    """
    mediapipe_to_openpose_map = {
        0: 0, 2: 16, 5: 15, 7: 18, 8: 17, 11: 5, 12: 2, 13: 6, 14: 3, 15: 7, 16: 4,
        23: 12, 24: 9, 25: 13, 26: 10, 27: 14, 28: 11, 29: 21, 30: 24, 31: 19, 32: 22
    }
    openpose_data = []
    for person_landmarks in mediapipe_landmarks_list:
        openpose_keypoints = np.zeros((25, 3), dtype=np.float32)
        for mp_idx, op_idx in mediapipe_to_openpose_map.items():
            if op_idx is not None and mp_idx < len(person_landmarks):
                landmark = person_landmarks[mp_idx]
                confidence = getattr(landmark, 'visibility', None) or getattr(landmark, 'presence', 0)
                openpose_keypoints[op_idx] = [landmark.x, landmark.y, confidence]
        openpose_keypoints_list = openpose_keypoints.tolist()
        openpose_data.append({"pose_keypoints_2d": openpose_keypoints_list})
    return openpose_data
