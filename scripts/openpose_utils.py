import os
import subprocess
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sktime.datasets import write_dataframe_to_tsfile
from angle_utils import AngleDataOpenPose
from angle_utils import AngleCalculatorOpenPose
from metadata_utils import parse_video_metadata


class OpenPoseConfig:
    """Configuration class for OpenPose parameters"""

    def __init__(self, openpose_build_path: str, model_dir: str, max_people: int = 1):
        self.openpose_build_path = openpose_build_path
        self.model_dir = model_dir
        self.max_people = max_people


class VideoProcessor:
    """Handles video processing with OpenPose"""

    def __init__(self, config: OpenPoseConfig):
        self.config = config

    def process_video(self, video_path: str, output_dir: str, json_filename: str) -> bool:
        """
        Process a video using OpenPose binary and save keypoints as JSON files
        """
        try:
            self._create_output_directories(output_dir)
            cmd = self._build_command(video_path, output_dir, json_filename)

            print(f"Processing video: {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error processing video: {result.stderr}")
                return False

            print(f"Successfully processed video. JSON output in: {os.path.join(output_dir, 'json')}")
            return True

        except Exception as e:
            print(f"Unexpected error processing video {video_path}: {e}")
            return False

    def _create_output_directories(self, output_dir: str) -> None:
        """Create necessary output directories"""
        os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
        # os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)

    def _build_command(self, video_path: str, output_dir: str, json_filename: str) -> List[str]:
        """Build the OpenPose command"""
        return [
            f"{self.config.openpose_build_path}/examples/openpose/openpose.bin",
            "--video", video_path,
            "--write_json", os.path.join(output_dir, "json", json_filename),
            "--display", "0",
            "--render_pose", "0",
            "--model_folder", self.config.model_dir,
            "--number_people_max", str(self.config.max_people)
        ]


class KeypointData:
    """Represents keypoint data for a single frame"""

    def __init__(self, keypoints: np.ndarray):
        self.keypoints = keypoints

    @classmethod
    def from_json(cls, json_file: str) -> 'KeypointData':
        """Create KeypointData from JSON file"""
        with open(json_file) as f:
            data = json.load(f)

        if not data.get("people"):
            return cls(np.zeros((25, 3), dtype=np.float32))

        keypoints = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32)
        keypoints = keypoints.reshape(-1, 3) #reformat so each row represents [x, y, confidence] for one body keypoint
        return cls(keypoints)


class VideoKeypointSequence:
    """Represents keypoint sequence for an entire video"""

    def __init__(self, video_id: str, keypoints_sequence: List[KeypointData]):
        self.video_id = video_id
        self.keypoints_sequence = keypoints_sequence

    @classmethod
    def from_json_directory(cls, json_directory: str, landmark_groups: Dict) -> Optional['VideoKeypointSequence']:
        """Create VideoKeypointSequence from directory of JSON files"""
        json_files = cls._get_sorted_json_files(json_directory)
        if not json_files:
            return None

        keypoints_sequence = []
        for json_file in json_files:
            print(f"Processing JSON file: {json_file}")
            keypoint_data = KeypointData.from_json(json_file)
            keypoint_with_angle = AngleDataOpenPose.from_keypoints(keypoint_data.keypoints, landmark_groups)
            keypoints_sequence.append(keypoint_with_angle)

        video_id = os.path.basename(os.path.normpath(json_directory))
        return cls(video_id, keypoints_sequence)

    @staticmethod
    def _get_sorted_json_files(directory: str) -> List[str]:
        """Get sorted list of JSON files from directory"""
        if not os.path.exists(directory):
            return []

        json_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".json")
        ]

        return sorted(json_files)


class TimeSeriesConverter:
    """Converts keypoint sequences to time series format"""

    def __init__(self):
        self.num_keypoints = 25
        self.coordinates = ("x", "y", "c")

    def convert_to_dataframe(self, video_sequence: VideoKeypointSequence, landmark_groups: Dict) -> pd.DataFrame:
        """
        Convert OpenPose sequence to sktime-compatible nested pandas DataFrame
        """
        if not video_sequence.keypoints_sequence:
            return pd.DataFrame()

        # Convert to 3D array (T, 25, 3)
        array_3d = self._sequence_to_3d_array(video_sequence.keypoints_sequence)

        # Flatten to 2D array (T, 75)
        flattened_array = self._flatten_keypoints(array_3d)

        # Create nested DataFrame
        return self._create_nested_dataframe(flattened_array, video_sequence.video_id, landmark_groups)

    def _sequence_to_3d_array(self, keypoints_sequence: List[KeypointData]) -> np.ndarray:
        """Convert list of KeypointData or ndarray to 3D numpy array"""
        arrays = []
        for kp in keypoints_sequence:
            if hasattr(kp, "keypoints"):
                arrays.append(kp.keypoints)
            elif isinstance(kp, np.ndarray):
                arrays.append(kp)
            else:
                raise TypeError(f"Unsupported sequence element type: {type(kp)}")

        return np.stack(arrays, axis=0) if arrays else np.empty((0, self.num_keypoints, len(self.coordinates)))

    def _flatten_keypoints(self, array_3d: np.ndarray) -> np.ndarray:
        """Flatten 3D keypoint array to 2D"""
        return array_3d.reshape(array_3d.shape[0], -1)

    def _create_nested_dataframe(self, flattened_array: np.ndarray, video_id: str, landmark_groups: Dict) -> pd.DataFrame:
        """Create nested pandas DataFrame in sktime format"""
        column_names = self._generate_column_names(landmark_groups)
        series_dict = self._create_series_dict(flattened_array, column_names)

        nested_df = pd.DataFrame([series_dict])
        nested_df.index = [video_id]

        return nested_df

    def _generate_column_names(self, landmark_groups: Dict) -> List[str]:
        """Generate column names for keypoint features"""

        column_names = [f"kp{kp_idx}_{coord}"
                for kp_idx in range(self.num_keypoints)
                for coord in self.coordinates]

        for joint_name in landmark_groups.keys():
            column_names.append(f"angle_{joint_name}")

        return column_names

    def _create_series_dict(self, flattened_array: np.ndarray, column_names: List[str]) -> Dict:
        """Create dictionary of pandas Series for nested DataFrame"""
        series_dict = {}
        for i, name in enumerate(column_names):
            series_dict[name] = pd.Series(flattened_array[:, i])
        return series_dict


class PoseDatasetBuilder:
    """Builds complete pose dataset from multiple videos"""

    def __init__(self, base_json_dir: str):
        self.base_json_dir = base_json_dir
        self.converter = TimeSeriesConverter()

    def build_dataset(self, landmark_groups: Dict, labels: np.ndarray) -> pd.DataFrame:
        """
        Build complete dataset from JSON files and metadata
        """
        video_directories = self._find_video_directories()
        nested_dfs = []
        counter = 1

        for video_dir in video_directories:
            print(f"\nProcessing video ({counter}/{len(video_directories)}): {video_dir}")

            video_df = self._process_single_video(video_dir, landmark_groups)
            if video_df is not None:
                nested_dfs.append(video_df)

                # parse name and add label to array
                video_name = video_dir.replace("poseEstKeypointsData/json/", "")
                parsed_details = parse_video_metadata("", video_name)
                correctness = parsed_details["correctness"]
                labels.append(correctness)

                print(f"Added to DataFrame: {video_dir} with label: {correctness}")

            counter += 1

        return pd.concat(nested_dfs, ignore_index=True) if nested_dfs else pd.DataFrame()

    def _find_video_directories(self) -> List[str]:
        """Find all video directories containing keypoint data"""
        if not os.path.exists(self.base_json_dir):
            return []

        return [
            os.path.join(self.base_json_dir, d) for d in os.listdir(self.base_json_dir)
            if os.path.isdir(os.path.join(self.base_json_dir, d)) and d.endswith('_keypoints')
        ]

    def _process_single_video(self, video_dir: str, landmark_groups: Dict) -> Optional[pd.DataFrame]:
        """Process a single video directory"""

        video_sequence = VideoKeypointSequence.from_json_directory(video_dir, landmark_groups)
        if video_sequence is None:
            return None

        return self.converter.convert_to_dataframe(video_sequence, landmark_groups)

class TSFileWriter:
    """Handles writing datasets to sktime .ts files"""

    @staticmethod
    def save_to_tsfile(combined_df: pd.DataFrame, labels: List, output_path: str,
                       problem_name: str = "PoseClassification") -> None:
        """
        Save a combined DataFrame in sktime's nested format to a .ts file.
        """
        if not isinstance(combined_df, pd.DataFrame):
            raise ValueError("The input must be a pandas DataFrame.")

        # Ensure labels are strings (required by write_dataframe_to_tsfile)
        str_labels = [str(label) for label in labels]

        write_dataframe_to_tsfile(
            data=combined_df,
            path=output_path,
            problem_name=problem_name,
            class_label=list(set(str_labels)),
            class_value_list=str_labels,
            comment="Dataset created from combined DataFrame",
            fold="_TRAIN"
        )

        print(f"Dataset saved to {output_path}/{problem_name}_TRAIN.ts")