import pandas as pd
import numpy as np
from sktime.datasets import write_dataframe_to_tsfile

def create_rocket_dataset(processed_data, labels, output_path, problem_name="PoseClassification"):
    """
    Create dataset in sktime format with proper length specification
    """
    if not isinstance(processed_data, list):
        processed_data = [processed_data]
    df_list = []
    for i, video_data in enumerate(processed_data):
        if video_data.ndim == 1:
            video_data = video_data.reshape(-1, 1)
        n_features = video_data.shape[1]
        row_dict = {}
        for j in range(n_features):
            row_dict[f"dim_{j}"] = pd.Series(video_data[:, j])
        df_list.append(row_dict)
    df = pd.DataFrame(df_list)
    lengths = [len(video_data) for video_data in processed_data]
    equal_length = len(set(lengths)) == 1
    series_length = lengths[0] if equal_length else None
    write_dataframe_to_tsfile(
        data=df,
        path=output_path,
        problem_name=problem_name,
        class_label=np.unique(labels).tolist(),
        class_value_list=labels,
        comment="OpenPose keypoints time series",
        fold="_TRAIN",
        equal_length=equal_length,
        series_length=series_length
    )
    print(f"Dataset saved to {output_path}")
