import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths to directories
audio_dir = "D:\\NLP_project\\svarah\\audio1"  # Replace with your correct audio files directory
transcriptions_csv = "transcriptions_no_duration.csv"  # Path to the CSV with audio_file and transcription
output_preprocessed_csv = "preprocessed_svarah_dataset.csv"  # Final output CSV

# Load transcriptions
transcriptions = pd.read_csv(transcriptions_csv)

# Ensure audio paths are correctly resolved
def resolve_audio_path(file_name):
    # Remove 'audio/' prefix if it exists
    file_name = file_name.replace("audio/", "")
    return os.path.join(audio_dir, file_name)

# Audio Normalization and Feature Extraction
def extract_audio_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=16000, mono=True)  # Normalize to 16 kHz, mono

        # Extract features
        duration = librosa.get_duration(y=audio, sr=sr)
        speech_rate = len(librosa.effects.split(audio)) / (duration + 1e-6)
        mean_pitch = np.mean(librosa.yin(audio, fmin=75, fmax=300, sr=sr))
        pause_count = sum([1 for frame in librosa.effects.split(audio) if len(frame) < sr * 0.2])
        articulation_rate = len(audio) / duration

        return {
            "duration": duration,
            "speech_rate": speech_rate,
            "mean_pitch": mean_pitch,
            "pause_count": pause_count,
            "articulation_rate": articulation_rate
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all audio files and extract features
audio_features = []
for file_name in tqdm(transcriptions["audio_file"]):
    file_path = resolve_audio_path(file_name)
    if os.path.exists(file_path):  # Ensure file exists
        features = extract_audio_features(file_path)
        if features:
            features["audio_file"] = file_name
            audio_features.append(features)
    else:
        print(f"File not found: {file_path}")

# Convert audio features to DataFrame
audio_features_df = pd.DataFrame(audio_features)

# Debug: Check if 'audio_file' column exists in audio_features_df
if "audio_file" not in audio_features_df.columns:
    raise KeyError("'audio_file' column is missing in audio features DataFrame!")

# Merge transcriptions with audio features
dataset = transcriptions.merge(audio_features_df, on="audio_file", how="inner")

# Save preprocessed dataset
dataset.to_csv(output_preprocessed_csv, index=False)

print(f"Preprocessing completed. Preprocessed dataset saved to {output_preprocessed_csv}")
