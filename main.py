from path_definitions import metadata_path, drawings_path
from data_loader import load_metadata, load_drawing_data
from feature_engineering import select_features
    

def main():
    metadata = load_metadata(metadata_path)
    print("Loaded metadata shape:", metadata.shape)

    print("Loading drawing data...")
    participants_data = load_drawing_data(metadata, drawings_path)
    print("Loaded drawing data for", len(participants_data), "participants")
    print("significant features: ", select_features(participants_data))
    # Add further processing here...

if __name__ == "__main__":
    main()