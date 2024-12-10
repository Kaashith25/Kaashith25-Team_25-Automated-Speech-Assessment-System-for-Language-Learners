import os
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import whisper
import librosa
import torch
from speechbrain.inference import Tacotron2, HIFIGAN

# Pretrained Pronunciation Model using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Sentiment Analysis Using RoBERTa
def analyze_sentiment(text):
    model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=-1).item()
    return "Negative" if sentiment == 0 else "Neutral" if sentiment == 1 else "Positive"

# Grammar Checking Using RoBERTa
def check_grammar(text):
    grammar_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = grammar_model(**inputs)
    grammar_score = torch.softmax(outputs.logits, dim=-1)
    return "Grammatically Correct" if grammar_score[0][1] > 0.5 else "Grammatically Incorrect"

# Fluency Analysis Using Librosa
def analyze_fluency(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=audio, sr=sr)
    silence = librosa.effects.split(audio, top_db=20)
    pauses = sum((end - start) / sr for start, end in silence)
    speech_rate = len(audio) / (sr * duration)  # Approx words per second
    return {"speech_rate": speech_rate, "pauses": pauses}

# Main Function
def analyze_speech(audio_path):
    print(f"Processing audio file: {audio_path}")
    
    # Transcription
    transcription = transcribe_audio(audio_path)
    print(f"Transcription: {transcription}")

    # Sentiment Analysis
    sentiment = analyze_sentiment(transcription)
    print(f"Sentiment: {sentiment}")

    # Grammar Check
    grammar = check_grammar(transcription)
    print(f"Grammar Check: {grammar}")

    # Fluency Analysis
    fluency_metrics = analyze_fluency(audio_path)
    print(f"Fluency Metrics: {fluency_metrics}")

    # Return all results
    return {
        "transcription": transcription,
        "sentiment": sentiment,
        "grammar": grammar,
        "fluency_metrics": fluency_metrics
    }

if __name__ == "__main__":
    audio_path = r"C:\Users\DELL\Downloads\nlp\src\ttsMP3.com_VoiceText_2024-12-9_0-6-16.wav"  # Replace with your audio file path
    results = analyze_speech(audio_path)
    print("Final Results:", results)
