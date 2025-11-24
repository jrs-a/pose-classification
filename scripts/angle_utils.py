import numpy as np
from typing import Dict, List, Tuple, Optional


class AngleCalculator3D:
    """
    A class to calculate 3D angles from pose landmarks using MediaPipe format.
    """

    def __init__(self, landmark_groups: Dict[str, List[int]] = None):
        """
        Initialize the AngleCalculator with landmark groups.

        Args:
            landmark_groups: Dictionary mapping joint names to lists of 3 landmark indices.
                           If None, uses DEFAULT_LANDMARK_GROUPS.
        """
        self.landmark_groups = landmark_groups
        self._validate_landmark_groups()

    def _validate_landmark_groups(self) -> None:
        """Validate that all landmark groups have exactly 3 points."""
        for name, indices in self.landmark_groups.items():
            if len(indices) != 3:
                raise ValueError(f"Landmark group '{name}' must have exactly 3 indices, got {len(indices)}")

    def extract_keypoints(self, pose_landmarks) -> Dict[str, np.ndarray]:
        """
        Extract (x, y, z) coordinates for all landmark groups.

        Args:
            pose_landmarks: MediaPipe pose landmarks object

        Returns:
            Dictionary mapping joint names to numpy arrays of shape (3, 3)
        """
        keypoints_dict = {}

        for keypoint_name, indices in self.landmark_groups.items():
            points = []
            for idx in indices:
                landmark = pose_landmarks.landmark[idx]
                points.append([landmark.x, landmark.y, landmark.z])

            keypoints_dict[keypoint_name] = np.array(points)

        return keypoints_dict

    @staticmethod
    def calculate_3d_angle(point_a: np.ndarray, point_mid: np.ndarray,
                           point_b: np.ndarray) -> float:
        """
        Calculate the 3D angle between three points.

        Args:
            point_a: First point (3D coordinates)
            point_mid: Middle point (vertex of the angle)
            point_b: Third point (3D coordinates)

        Returns:
            Angle in degrees between vectors (point_a->point_mid) and (point_b->point_mid)
        """
        # Calculate vectors
        vector_a = point_a - point_mid
        vector_b = point_b - point_mid

        # Calculate dot product and magnitudes
        dot_product = np.dot(vector_a, vector_b)
        mag_a = np.linalg.norm(vector_a)
        mag_b = np.linalg.norm(vector_b)

        # Handle division by zero
        if mag_a == 0 or mag_b == 0:
            return 0.0

        # Calculate cosine of the angle
        cos_theta = np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0)

        # Convert to degrees
        angle_radians = np.arccos(cos_theta)
        return np.degrees(angle_radians)

    def process_pose(self, pose_landmarks, verbose: bool = False) -> Dict[str, float]:
        """
        Process pose landmarks and calculate all angles.

        Args:
            pose_landmarks: MediaPipe pose landmarks object
            verbose: Whether to print debug information

        Returns:
            Dictionary mapping joint names to angles in degrees
        """
        if pose_landmarks is None:
            raise ValueError("No pose landmarks provided")

        # Extract keypoints
        # keypoints_dict = self.extract_keypoints(pose_landmarks) -> only for mediapipe
        keypoints_dict = dict(np.ndenumerate(pose_landmarks))
        angle_dict = {}

        # Calculate angles for each joint group
        for joint_name, points in keypoints_dict.items():
            if verbose:
                print(f"{joint_name}:")
                print(f"  Point A: {points[0]}")
                print(f"  Midpoint: {points[1]}")
                print(f"  Point B: {points[2]}")

            angle = self.calculate_3d_angle(points[0], points[1], points[2])
            angle_dict[joint_name] = angle

            if verbose:
                print(f"  Angle: {angle:.2f}°\n")

        return angle_dict

    def get_available_joints(self) -> List[str]:
        """Get list of available joint names that can be analyzed."""
        return list(self.landmark_groups.keys())

class AngleData:
    """Represents angle data for a single frame"""

    def __init__(self, keypoints: np.ndarray):
        self.keypoints = keypoints

    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray, landmark_groups: Dict) -> 'AngleData':
        """Create AngleData from keypoints"""

        angle_calculator = AngleCalculator3D(landmark_groups)
        angle_dict = angle_calculator.process_pose(keypoints, verbose=True)

        for joint_name, angle in angle_dict.items():
            new_angle = np.array([[joint_name], [angle]], dtype=np.float32)
            np.hstack(keypoints, new_angle)

        return cls(keypoints)