# =====================================
# Install dependencies first (run once):
# pip install torch torchaudio transformers sounddevice librosa soundfile
# =====================================

import torch
import sounddevice as sd
import librosa
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ===============================
# 1. Load Pretrained Models
# ===============================
MODELS = {
    "Mostaffa": "Zeyadd-Mostaffa/Deepfake-Audio-Detection-v1",
    # "Fine-Tuned Model": r"C:\Users\SHABBIR\Desktop\major\fine_tuned_asvspoof_la"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_models = {}
for name, model_id in MODELS.items():
    print(f"Loading model: {name} ({model_id}) ...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)
    loaded_models[name] = (feature_extractor, model)

print("✅ Models loaded successfully!")


# ===============================
# 2. Record Audio from Mic
# ===============================
def record_audio(filename="voice.wav", duration=5, sr=16000):
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()  # wait until recording finishes
    sf.write(filename, audio, sr)
    print(f"✅ Saved recording to {filename}")
    return filename


# ===============================
# 3. Preprocess Audio
# ===============================
def preprocess_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio / np.max(np.abs(audio))
    return audio, target_sr


# ===============================
# 4. Deepfake Detection
# ===============================
def detect_deepfake(file_path, threshold=0.7):
    results = {}
    for name, (processor, model) in loaded_models.items():
        audio, sr = preprocess_audio(file_path)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs.to(device)).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        fake_prob = float(probs[1])  # index 1 = fake
        results[name] = {
            "Fake Probability": round(fake_prob, 4),
            "Prediction": "Fake" if fake_prob > threshold else "Real"
        }
    return results


# ===============================
# 5. Run Everything
# ===============================
if __name__ == "__main__":
    file_path = record_audio(duration=5)  # record your voice
    results = detect_deepfake(file_path, threshold=0.7)

    print("\n🔎 Detection Results:")
    for model_name, outcome in results.items():
        print(f"{model_name}: {outcome}")
