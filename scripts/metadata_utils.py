import os

def parse_video_metadata(root, file):
    try:
        name, ext = os.path.splitext(file)
        parts = name.split('_')
        exercise, view, correctness, participant, unique_id = parts
        return {
            "file_name": file,
            "exercise": exercise,
            "view": view,
            "correctness": correctness,
            "participant": participant,
            "unique_id": unique_id,
            "drive_path": os.path.join(root, file),
            "extension": ext
        }
    except Exception as e:
        print(f"Skipping {file}: {str(e)}")
        return None