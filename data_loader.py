"""
Module: data_loader.py
Description: Functions to load and preprocess PaHaW data.
"""

import os
import pandas as pd
import numpy as np
import os


# You may want to parameterize these in main.py instead of hardcoding
def load_metadata(metadata_path):
    return pd.read_excel(metadata_path)
    
def read_svc_file(filepath):
    """
    Reads a .svc file and returns stroke data.
    Dummy implementation - replace with your actual parsing logic.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        points = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])
        return points

def load_drawing_data(metadata, drawings_path):
    """
    Loads drawing data and metadata for each participant.
    """
    participants_data = []

    for _, row in metadata.iterrows():
        participant_id = f"{row['ID']:05d}"
        participant_info = row.to_dict()
        participant_drawings = []

        for drawing_num in range(1, 9):
            svc_file_path = os.path.join(drawings_path, participant_id, f"{participant_id}__{drawing_num}_1.svc")
            if os.path.exists(svc_file_path):
                drawing_data = read_svc_file(svc_file_path)
                drawing_data = np.array(drawing_data)
                participant_drawings.append(drawing_data)

        if len(participant_drawings) == 8:
            participant_info['drawings'] = participant_drawings
            participants_data.append(participant_info)

    return participants_data




# Example combined loading function if applicable
def load_all_data(metadata_path, drawings_path):
    metadata = load_metadata(metadata_path)
    drawing_data = load_drawing_data(drawings_path)
    # Combine metadata and drawings as needed
    return metadata, drawing_data