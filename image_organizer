import os
import pandas as pd

# Updated dictionary for subject metadata (add real entries as needed)
subject_dict = {
    "546": {"ID": 546, "Visits": [21, 45], "Publishing": "Yes"},
    "732": {"ID": 732, "Visits": [12, 38], "Publishing": "No"},
    # Add more subject metadata here
}

# Root directory where the dataset is stored (example: ODINE_ALIGNED)
dataset_path = "/mnt/data/ODINE_ALIGNED"

# List to store extracted data
data_list = []

# Function to determine metadata from file or path
def extract_metadata(file_path):
    file_path_lower = file_path.lower()
    condition = "Neutral"
    if "expression" in file_path_lower:
        condition = "Expression"
    elif any(x in file_path_lower for x in ["frontal", "pitch", "yaw"]):
        condition = "Neutral"

    eyewear = "Yes" if "eyeglasses" in file_path_lower else "No"
    mask = "Yes" if "mask" in file_path_lower else "No"
    consent = "Yes" if "consent" in file_path_lower else "No"
    destructors = "Yes" if "destructor" in file_path_lower else "No"
    mode = "Thermal" if "thermal" in file_path_lower else "Broadband"

    return condition, eyewear, mask, consent, destructors, mode

# Traverse the directory tree
for mode_folder in ["visible", "thermal"]:
    mode_path = os.path.join(dataset_path, mode_folder)
    if not os.path.isdir(mode_path):
        continue

    for day_folder in os.listdir(mode_path):
        day_path = os.path.join(mode_path, day_folder)
        if not os.path.isdir(day_path):
            continue

        for subfolder in os.listdir(day_path):
            subfolder_path = os.path.join(day_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for image_file in os.listdir(subfolder_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                    file_path = os.path.join(subfolder_path, image_file)

                    # Attempt to extract subject ID from filename
                    subject_id_key = image_file.split("_")[0].replace("Subject_", "")
                    subject_info = subject_dict.get(subject_id_key, {"ID": "Unknown", "Publishing": "Unknown"})
                    subject_id = subject_info["ID"]
                    publishing_status = subject_info["Publishing"]

                    # Extract metadata
                    condition, eyewear, mask, consent, destructors, mode = extract_metadata(file_path)

                    # Append data
                    data_list.append([
                        subject_id, file_path, condition, eyewear, mask, consent, destructors, mode, publishing_status
                    ])

# Convert to DataFrame
df = pd.DataFrame(data_list, columns=[
    "Subject ID", "File Path", "Condition", "Eyewear", "Mask", "Consent", "Destructors", "Mode", "Publishing"
])

# Save CSV
csv_output_path = "/mnt/data/dataset_metadata.csv"
df.to_csv(csv_output_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Dataset Metadata", dataframe=df)
csv_output_path
