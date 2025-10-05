import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import interpolate

def preprocess_keypoints(sequence):
    """
    Preprocess keypoints for time series classification:
    1. Handle missing values
    2. Normalize coordinates
    3. Compute velocities (1st derivative)
    4. Flatten features per frame
    """
    conf_threshold = 0.2
    for i in range(len(sequence)):
        frame = sequence[i]
        low_conf = frame[:, 2] < conf_threshold
        frame[low_conf, :2] = np.nan
    for k in range(25):
        x = sequence[:, k, 0]
        y = sequence[:, k, 1]
        valid = ~np.isnan(x)
        indices = np.arange(len(x))
        if np.sum(valid) > 1:
            x_interp = interpolate.interp1d(indices[valid], x[valid], bounds_error=False, fill_value="extrapolate")
            y_interp = interpolate.interp1d(indices[valid], y[valid], bounds_error=False, fill_value="extrapolate")
            sequence[:, k, 0] = x_interp(indices)
            sequence[:, k, 1] = y_interp(indices)
    for i in range(len(sequence)):
        frame = sequence[i]
        non_zero = frame[:, 2] > conf_threshold
        if np.any(non_zero):
            min_x, min_y = np.min(frame[non_zero, :2], axis=0)
            max_x, max_y = np.max(frame[non_zero, :2], axis=0)
            width = max_x - min_x
            height = max_y - min_y
            if width > 0 and height > 0:
                frame[:, 0] = (frame[:, 0] - min_x) / width
                frame[:, 1] = (frame[:, 1] - min_y) / height
    velocities = np.diff(sequence[:, :, :2], axis=0)
    velocities = np.pad(velocities, ((0, 1), (0, 0), (0, 0)))
    combined = np.concatenate([
        sequence[:, :, :2],
        velocities,
        sequence[:, :, [2]]
    ], axis=2)
    flattened = combined.reshape(combined.shape[0], -1)
    if flattened.ndim == 1:
        flattened = flattened.reshape(1, -1)
    return flattened
