import os
import subprocess
import json
import numpy as np

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

def json_to_time_series(json_dir):
    """
    Convert OpenPose JSON output to time series array
    Format: (frames, keypoints, 3) where 3 = [x, y, confidence]
    """
    json_files = sorted([
        os.path.join(json_dir, f) for f in os.listdir(json_dir)
        if f.endswith(".json")
    ])
    keypoints_sequence = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            if not data["people"]:
                keypoints_sequence.append(np.zeros((25, 3)))
                continue
            keypoints = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32)
            keypoints = keypoints.reshape(-1, 3)
            keypoints_sequence.append(keypoints)
    return np.array(keypoints_sequence)

def build_dataframe(dataset, base_json_dir):
    """
    Process JSON files and include metadata from the original dataset,
    returning a pandas DataFrame
    """
    import pandas as pd
    train_dataset = dataset['train']
    all_rows = []
    video_metadata = {}
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        base_name = os.path.splitext(os.path.basename(item['drive_path']))[0]
        video_metadata[base_name] = item
    video_dirs = [
        d for d in os.listdir(base_json_dir)
        if os.path.isdir(os.path.join(base_json_dir, d)) and d.endswith('_keypoints')
    ]
    for video_dir in video_dirs:
        video_path = os.path.join(base_json_dir, video_dir)
        video_id = video_dir.replace('_keypoints', '')
        metadata = video_metadata.get(video_id, {})
        print(f"\nProcessing: {video_id}")
        time_series = json_to_time_series(video_path)
        if time_series is not None:
            if isinstance(time_series, np.ndarray):
                time_series = time_series.tolist()
            row = {
                'video_id': video_id,
                'time_series': time_series
            }
            row.update(metadata)
            all_rows.append(row)
            print(f"Added to DataFrame: {video_id}")
    return pd.DataFrame(all_rows)
