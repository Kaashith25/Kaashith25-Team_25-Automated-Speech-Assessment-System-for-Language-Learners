# Automated Speech Assessment System for Language Learners (Initial Trials)

## Introduction
The **Automated Speech Assessment System for Language Learners** aims to evaluate spoken language proficiency and fluency using a combination of acoustic features and natural language processing (NLP) techniques. The project is designed to provide an objective and comprehensive assessment of speech characteristics, including mood, pronunciation, articulation, and fluency. This README details the initial trials, focusing on data collection, feature extraction, model development, and challenges faced during implementation.

## Approach Overview
In this phase, the project workflow was designed as follows:
1. **Data Collection**: Audio recordings and associated text data were obtained from multiple datasets and resources, including the Svarah Dataset, CEFR Dataset, and additional audio extracted from YouTube TED Talks.
2. **Feature Engineering**: Linguistic and acoustic features were extracted from speech and transcriptions for training the models.
3. **Model Development**: Two machine learning models were built:
   - A classification model for proficiency level prediction.
   - A regression model for fluency scoring.

## Development Process
### Initial Data Collection and Challenges
Initially, we implemented a Python script using `yt-dlp` to download audio from selected TED Talks on YouTube. While this approach provided additional audio samples for analysis, it introduced significant challenges:
- Variability in audio quality across TED Talks affected consistency.
- Manual curation of URLs and handling metadata proved inefficient.
- The downloaded data was not aligned with the structure of the primary datasets (Svarah and CEFR), leading to integration difficulties.

Due to these issues, this approach was discontinued, and the focus shifted exclusively to the **Svarah Dataset** and **CEFR Dataset** for subsequent steps.

### Dataset Preparation
1. **Preprocessing Svarah Dataset**:
   - Transcriptions were cleaned, and audio features were extracted.
   - The dataset was standardized to remove noise and inconsistencies.

2. **Integrating with CEFR Dataset**:
   - The CEFR Dataset provided additional annotations such as proficiency levels.
   - Combining these datasets involved mapping features effectively to align with CEFR standards.

### Feature Extraction
We focused on extracting features critical to evaluating speech proficiency:
- Acoustic Features: Pitch (f0), formants (f1 and f2), speech rate, articulation rate, and pause durations.
- Linguistic Features: Number of words, filler words, and transcription-based metrics.
- Semantic Analysis: Sentiment and mood detection using basic NLP tools like `TextBlob`.

### Model Training
1. **Training Data**:
   - The combined dataset was split into training, validation, and testing subsets.
   - Feature vectors and corresponding labels were prepared for model input.

2. **Models**:
   - **Proficiency Level Classification Model**: Predicts the speaker's language proficiency category.
   - **Fluency Score Regression Model**: Assigns a numerical score to the speaker's fluency.

### Challenges and Failure Analysis
The initial trials encountered significant challenges that hindered the success of the models:
1. **Dataset Imbalance**:
   - Some proficiency levels were underrepresented, leading to biased model predictions.
   
2. **Feature Insufficiency**:
   - Extracted features were insufficient to capture the intricacies of speech. For example, nuanced aspects of fluency, such as tone and rhythm, were not fully represented.

3. **Model Performance**:
   - Models failed to generalize effectively, likely due to overfitting and limited feature diversity.
   - Simplistic modeling approaches were inadequate for the complexity of speech analysis tasks.

4. **Evaluation**:
   - The models produced inconsistent predictions, failing to meet the desired standards of accuracy.

## Evaluation and Results
Despite comprehensive preprocessing and model training efforts, the results highlighted the limitations of the chosen approach:
- The proficiency classification model struggled to differentiate effectively between levels.
- The fluency regression model provided inconsistent scores, reflecting the limitations in feature engineering and model selection.

These results demonstrated that the approach required significant refinement to achieve the project goals.

## Conclusion
The initial trials served as a valuable learning experience, revealing critical areas for improvement:
1. Moving forward, enhanced feature extraction techniques must incorporate advanced acoustic and linguistic metrics.
2. The inclusion of more sophisticated machine learning models, such as transformer-based models for semantic analysis, may help address the complexity of the task.
3. Addressing dataset imbalances and ensuring consistent data quality are essential for improving model performance.

While these initial trials did not meet expectations, they provided the foundation for refining our project in future iterations.