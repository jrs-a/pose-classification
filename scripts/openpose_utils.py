import os
import subprocess
import json
import numpy as np
import pandas as pd
from sktime.datasets import write_dataframe_to_tsfile

# OpenPose video processing
def process_video_with_openpose(video_path, output_dir, model_dir, json_filename, OPENPOSE_BUILD_PATH):
    """
    Process a video using OpenPose binary and save keypoints as JSON files
    """
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)
    cmd = [
        f"{OPENPOSE_BUILD_PATH}/examples/openpose/openpose.bin",
        "--video", video_path,
        "--write_json", os.path.join(output_dir, "json", json_filename),
        "--display", "0",
        "--render_pose", "0",
        "--model_folder", model_dir,
        "--number_people_max", "1"
    ]
    print(f"Processing video: {video_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error processing video: {result.stderr}")
        return False
    print(f"Successfully processed video. JSON output in: {os.path.join(output_dir, 'json')}")
    return True

def json_to_time_series(video_path):
    """
    Convert OpenPose JSON output to an sktime-compatible nested pandas DataFrame (one-row panel).

    Input:
    - video_path: path to a directory containing OpenPose JSON files for the video
      (one JSON per frame). Filenames will be sorted lexicographically to form the
      temporal order.

    Output:
    - pandas.DataFrame with one row (one series/sample). Each column corresponds
      to a single dimension (keypoint_x/y/conf for each of the 25 keypoints,
      total 25*3 = 75 columns). Each cell contains a pandas.Series of length T
      (number of frames) representing the time series for that dimension. This
      nested DataFrame is the canonical sktime panel format for a single
      multivariate time series sample.

    Notes / assumptions:
    - If a frame has no detected people, a (25,3) array of zeros is used for that frame.
    - Columns are named kp{idx}_{coord}, where coord is x, y, or c (confidence).
    """
    json_files = sorted([
        os.path.join(video_path, f) for f in os.listdir(video_path)
        if f.endswith(".json")
    ])
    keypoints_sequence = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            if not data.get("people"):
                keypoints_sequence.append(np.zeros((25, 3), dtype=np.float32))
                continue
            keypoints = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32)
            keypoints = keypoints.reshape(-1, 3)
            keypoints_sequence.append(keypoints)

    if not keypoints_sequence:
        # return empty DataFrame if no frames found
        return pd.DataFrame()

    arr = np.stack(keypoints_sequence, axis=0)  # shape (T, 25, 3)
    T = arr.shape[0]
    # flatten last two dims to obtain 75 separate dimensions: order kp0_x,kp0_y,kp0_c,kp1_x,...
    dims = arr.reshape(T, -1)  # shape (T, 75)

    # build nested DataFrame: one row, columns for each flattened dim, cell contains pd.Series
    col_names = []
    for kp_idx in range(arr.shape[1]):
        for coord in ("x", "y", "c"):
            col_names.append(f"kp{kp_idx}_{coord}")

    series_dict = {}
    for i, name in enumerate(col_names):
        # create a pandas Series for each dimension with integer index 0..T-1
        series_dict[name] = pd.Series(dims[:, i])

    nested_df = pd.DataFrame([series_dict])
    # set a meaningful index (directory basename) if possible
    try:
        nested_df.index = [os.path.basename(os.path.normpath(video_path))]
    except Exception:
        pass

    return nested_df

def build_dataframe(dataset, base_json_dir):
    """
    Process JSON files and include metadata from the original dataset,
    returning a combined pandas DataFrame in sktime's nested format.

    Parameters:
    - dataset: dictionary containing metadata for the dataset.
    - base_json_dir: directory containing JSON files for videos.

    Returns:
    - pandas.DataFrame in sktime's nested format, with metadata included.
    """
    train_dataset = dataset['train']
    video_metadata = {}
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        base_name = os.path.splitext(os.path.basename(item['drive_path']))[0]
        video_metadata[base_name] = item

    video_dirs = [
        d for d in os.listdir(base_json_dir)
        if os.path.isdir(os.path.join(base_json_dir, d)) and d.endswith('_keypoints')
    ]

    nested_dfs = []  # Collect nested DataFrames for combination

    for video_dir in video_dirs:
        video_path = os.path.join(base_json_dir, video_dir)
        video_id = video_dir.replace('_keypoints', '')
        metadata = video_metadata.get(video_id, {})
        print(f"\nProcessing: {video_id}")
        time_series = json_to_time_series(video_path)
        if time_series is not None:
            nested_dfs.append(time_series)  # Add to nested DataFrame list
            print(f"Added to DataFrame: {video_id}")

    # Combine all nested DataFrames into a single DataFrame
    combined_df = pd.concat(nested_dfs, ignore_index=True)

    return combined_df

def result_df_to_tsfile(combined_df, labels, output_path, problem_name="PoseClassification"):
    """
    Save a combined DataFrame in sktime's nested format to a .ts file.

    Parameters:
    - combined_df: pandas.DataFrame in sktime's nested format.
    - labels: list of class labels corresponding to each row in the DataFrame.
    - output_path: directory to save the .ts file.
    - problem_name: name of the classification problem (used in the .ts file).
    """
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")

    # Ensure labels are strings (required by write_dataframe_to_tsfile)
    labels = [str(label) for label in labels]

    write_dataframe_to_tsfile(
        data=combined_df,
        path=output_path,
        problem_name=problem_name,
        class_label=list(set(labels)),
        class_value_list=labels,
        comment="Dataset created from combined DataFrame",
        fold="_TRAIN"
    )

    print(f"Dataset saved to {output_path}/{problem_name}_TRAIN.ts")
