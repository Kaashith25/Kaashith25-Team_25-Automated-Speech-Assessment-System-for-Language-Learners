
Automated Speech Assessment System for Language Learners

This project evaluates spoken language proficiency by analyzing transcription, sentiment, grammar, and fluency using advanced NLP models and audio processing tools.

---

Setup Instructions:
1. Ensure Python 3.x is installed on your system.
2. Download and install `ffmpeg`, and set its binary path in your environment:
   Example:
os.environ["FFMPEG_BINARY"] = r"C:\path\to\ffmpeg.exe"
3. Install the required Python libraries by running:
pip install -r requirements.txt
The `requirements.txt` includes:
- transformers
- torch
- librosa
- speechbrain
- openai-whisper

---

How to Run the Script:
1. Open `main.py` and update the `audio_path` variable with the full path to your audio file.
Example:
audio_path = r"C:\path\to\your\audio.wav"
2. Run the script in your terminal or command prompt:
python main.py

---

What the Script Does:
1. Transcribes audio into text using the OpenAI Whisper model.
2. Analyzes the transcription for sentiment using a pre-trained RoBERTa model.
3. Checks the grammar of the transcription.
4. Extracts fluency metrics like speech rate and pause duration.

---

Expected Output:
The script will print and return the following results:
- Transcription of the audio file.
- Sentiment analysis result: Positive, Neutral, or Negative.
- Grammar evaluation: Grammatically Correct or Incorrect.
- Fluency metrics including speech rate and total pauses.

---

Project Notes:
- Make sure the audio file is in `.wav` format for best compatibility.
- If you encounter any issues, refer to the comments within the script for clarification.
