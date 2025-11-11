"""
Deepfake Audio Detection Streamlit App
Real-time detection of AI-generated voices from microphone input
"""

import streamlit as st
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMainBlockContainer {
        padding-top: 2rem;
    }
    .header-title {
        text-align: center;
        color: #ff6b6b;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .result-genuine {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .result-deepfake {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

@st.cache_resource
def load_model(model_path="Zeyadd-Mostaffa/Deepfake-Audio-Detection-v1"):
    """Load model once and cache it"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return feature_extractor, model, device

def preprocess_audio(audio_data, sr=16000, target_sr=16000):
    """Preprocess audio for model input"""
    try:
        # Convert to mono if stereo
        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        # Normalize
        maxv = np.max(np.abs(audio_data))
        if maxv > 0:
            audio_data = audio_data / maxv
        
        return audio_data
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None

def detect_deepfake(audio_data, sr, feature_extractor, model, device):
    """Detect if audio is deepfake"""
    try:
        # Preprocess
        audio = preprocess_audio(audio_data, sr=sr)
        if audio is None:
            return None
        
        # Extract features
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        genuine_prob = float(probs[0])
        deepfake_prob = float(probs[1])
        
        return {
            "genuine_probability": genuine_prob,
            "deepfake_probability": deepfake_prob,
            "is_deepfake": deepfake_prob > 0.5
        }
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None

def audio_callback(frame):
    """Callback for WebRTC audio frames"""
    audio = frame.to_ndarray()
    return av.AudioFrame.from_ndarray(audio, format="s16", layout="mono")

# Header
st.markdown('<div class="header-title">🎙️ Deepfake Audio Detector</div>', unsafe_allow_html=True)
st.markdown("### Real-time Detection of AI-Generated Voices")
st.divider()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    model_path = st.text_input(
        "Model Path",
        value="./fine_tuned_asvspoof_la",
        help="Path to fine-tuned model"
    )
    
    detection_mode = st.radio(
        "Detection Mode",
        ["Live Microphone", "Upload File", "WebRTC Stream"],
        help="Choose how to input audio"
    )
    
    detection_threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower = more sensitive to deepfakes"
    )
    
    st.divider()
    st.info(
        "📊 **Model Info:**\n\n"
        "- EER: 0.33% (Excellent)\n"
        "- Detects: Voice synthesis, conversion, deepfakes\n"
        "- Sample rate: 16kHz\n"
        "- Training data: ASVspoof2019 LA"
    )

# Load model
try:
    feature_extractor, model, device = load_model(model_path)
    st.session_state.model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.session_state.model_loaded = False

if st.session_state.model_loaded:
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if detection_mode == "Live Microphone":
            st.subheader("🎤 Live Microphone Input")
            
            audio_data = st.audio_input(
                "Record audio from your microphone",
                help="Click to record audio, then click again to stop"
            )
            
            if audio_data is not None:
                # Convert audio bytes to numpy array
                audio_bytes = audio_data.getbuffer()
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Perform detection
                with st.spinner("🔍 Analyzing audio..."):
                    result = detect_deepfake(
                        audio_array,
                        sr=16000,
                        feature_extractor=feature_extractor,
                        model=model,
                        device=device
                    )
                
                if result:
                    # Display result
                    is_deepfake = result["deepfake_probability"] > detection_threshold
                    
                    if is_deepfake:
                        st.markdown(
                            f"""
                            <div class="result-deepfake">
                            <h3>⚠️ DEEPFAKE DETECTED</h3>
                            <p>This audio appears to be AI-generated or synthetic.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-genuine">
                            <h3>✅ GENUINE AUDIO</h3>
                            <p>This audio appears to be real/authentic.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Metrics
                    col_gen, col_deep = st.columns(2)
                    with col_gen:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <h4>Genuine (Real)</h4>
                            <h2 style="color: #28a745;">{result['genuine_probability']*100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_deep:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <h4>Deepfake (AI-Generated)</h4>
                            <h2 style="color: #dc3545;">{result['deepfake_probability']*100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Add to history
                    history_item = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mode": "Live",
                        "Genuine %": f"{result['genuine_probability']*100:.1f}",
                        "Deepfake %": f"{result['deepfake_probability']*100:.1f}",
                        "Result": "Deepfake" if is_deepfake else "Genuine"
                    }
                    st.session_state.detection_history.append(history_item)
        
        elif detection_mode == "Upload File":
            st.subheader("📁 Upload Audio File")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "flac", "ogg"],
                help="Upload WAV, MP3, FLAC, or OGG files"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load audio
                audio_data, sr = librosa.load(temp_path, sr=None)
                
                # Play audio
                st.audio(temp_path)
                
                # Perform detection
                with st.spinner("🔍 Analyzing audio..."):
                    result = detect_deepfake(
                        audio_data,
                        sr=sr,
                        feature_extractor=feature_extractor,
                        model=model,
                        device=device
                    )
                
                if result:
                    # Display result
                    is_deepfake = result["deepfake_probability"] > detection_threshold
                    
                    if is_deepfake:
                        st.markdown(
                            f"""
                            <div class="result-deepfake">
                            <h3>⚠️ DEEPFAKE DETECTED</h3>
                            <p>This audio appears to be AI-generated or synthetic.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-genuine">
                            <h3>✅ GENUINE AUDIO</h3>
                            <p>This audio appears to be real/authentic.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Metrics
                    col_gen, col_deep = st.columns(2)
                    with col_gen:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <h4>Genuine (Real)</h4>
                            <h2 style="color: #28a745;">{result['genuine_probability']*100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_deep:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                            <h4>Deepfake (AI-Generated)</h4>
                            <h2 style="color: #dc3545;">{result['deepfake_probability']*100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Add to history
                    history_item = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mode": "Upload",
                        "File": uploaded_file.name,
                        "Genuine %": f"{result['genuine_probability']*100:.1f}",
                        "Deepfake %": f"{result['deepfake_probability']*100:.1f}",
                        "Result": "Deepfake" if is_deepfake else "Genuine"
                    }
                    st.session_state.detection_history.append(history_item)
                
                # Clean up
                Path(temp_path).unlink()
        
        elif detection_mode == "WebRTC Stream":
            st.subheader("📡 WebRTC Real-Time Stream")
            
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            webrtc_ctx = webrtc_streamer(
                key="deepfake-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
                title="🎙️ WebRTC Stream"
            )
            
            if webrtc_ctx.state.playing:
                st.info("🔴 Recording... WebRTC streaming is active")
    
    with col2:
        st.subheader("📊 Detection History")
        if st.session_state.detection_history:
            df = pd.DataFrame(st.session_state.detection_history)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.divider()
            deepfake_count = sum(1 for item in st.session_state.detection_history if item["Result"] == "Deepfake")
            genuine_count = len(st.session_state.detection_history) - deepfake_count
            
            st.metric("Total Detections", len(st.session_state.detection_history))
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Deepfakes", deepfake_count)
            with col_b:
                st.metric("Genuine", genuine_count)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.detection_history = []
                st.rerun()
        else:
            st.info("No detections yet. Start analyzing audio!")

else:
    st.error("Model failed to load. Please check the model path.")