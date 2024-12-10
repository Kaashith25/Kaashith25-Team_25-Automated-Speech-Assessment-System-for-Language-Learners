import pandas as pd
import numpy as np
from collections import Counter
import math

# Paths
cefr_csv = r"D:\NLP_project\cefr\cefr_dataset.csv"  # Path to CEFR dataset
svarah_preprocessed_csv = r"D:\NLP_project\svarah\preprocessed_svarah_dataset.csv"  # Preprocessed Svarah dataset
output_dataset_csv = "svarah_with_cefr_features.csv"  # Output file

# Load datasets
cefr_data = pd.read_csv(cefr_csv)
svarah_data = pd.read_csv(svarah_preprocessed_csv)

# Normalize CEFR dataset
cefr_data['headword'] = cefr_data['headword'].str.lower()  # Convert words to lowercase
cefr_mapping = cefr_data.set_index('headword')['CEFR'].to_dict()  # Create a word-to-CEFR mapping

# Define CEFR level encoding for calculations
cefr_encoding = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# Helper function: Calculate lexical features
def calculate_lexical_features(transcription):
    words = transcription.lower().split()  # Tokenize transcription into words
    cefr_levels = []
    for word in words:
        cefr_level = cefr_mapping.get(word, "Unknown")
        if cefr_level != "Unknown":
            cefr_levels.append(cefr_encoding[cefr_level])

    if not cefr_levels:
        # No known words, return zeros
        return {
            "average_cefr": 0,
            "percentage_advanced": 0,
            "lexical_diversity": 0
        }

    # Calculate average CEFR level
    average_cefr = sum(cefr_levels) / len(cefr_levels)

    # Calculate percentage of advanced words (B2+)
    advanced_count = sum(1 for level in cefr_levels if level >= cefr_encoding['B2'])
    percentage_advanced = (advanced_count / len(cefr_levels)) * 100

    # Calculate lexical diversity (Shannon Index)
    level_counts = Counter(cefr_levels)
    total_count = len(cefr_levels)
    lexical_diversity = -sum((count / total_count) * math.log(count / total_count) for count in level_counts.values())

    return {
        "average_cefr": average_cefr,
        "percentage_advanced": percentage_advanced,
        "lexical_diversity": lexical_diversity
    }

# Apply lexical feature extraction
lexical_features = svarah_data['transcription'].apply(calculate_lexical_features)

# Convert features to DataFrame and merge with Svarah dataset
lexical_features_df = pd.DataFrame(list(lexical_features))
svarah_with_cefr = pd.concat([svarah_data, lexical_features_df], axis=1)

# Save updated dataset
svarah_with_cefr.to_csv(output_dataset_csv, index=False)

print(f"Dataset updated with CEFR features. Saved to {output_dataset_csv}.")
