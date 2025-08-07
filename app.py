import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import tempfile

# Modern arka plan ve stil ekleme
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1a1a40 0%, #2d2d7a 50%, #ff3cac 100%);
        min-height: 100vh;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #1a1a40 0%, #2d2d7a 50%, #ff3cac 100%);
        min-height: 100vh;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        background: radial-gradient(circle at 20% 30%, #00f2fe55 0%, transparent 70%),
                    radial-gradient(circle at 80% 70%, #ff3cac55 0%, transparent 70%),
                    repeating-radial-gradient(circle at 50% 50%, #fff2 0, #fff2 1px, transparent 2px, transparent 100px);
        opacity: 0.7;
        pointer-events: none;
    }
    
    /* Sidebar stilini özelleştirme */
    [data-testid="stSidebar"] {
        background-color: rgba(26, 26, 64, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buton stilini özelleştirme */
    .stButton > button {
        background: linear-gradient(45deg, #00f2fe, #ff3cac);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 242, 254, 0.3);
    }
    
    /* Başlık stilini özelleştirme */
    h1 {
        background: linear-gradient(to right, #00f2fe, #ff3cac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em !important;
        font-weight: bold !important;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Metin rengini ayarlama */
    .stMarkdown {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dalga efekti SVG
st.markdown(
    '''
    <svg width="100%" height="150" viewBox="0 0 1920 150" fill="none" xmlns="http://www.w3.org/2000/svg" style="position:absolute;top:0;left:0;z-index:-1;">
      <path d="M0 0H1920V150H0V0Z" fill="url(#paint0_linear)"/>
      <path d="M0 100C400 200 800 0 1200 100C1600 200 1920 0 1920 0V150H0V100Z" fill="url(#paint1_radial)"/>
      <defs>
        <linearGradient id="paint0_linear" x1="0" y1="0" x2="1920" y2="150" gradientUnits="userSpaceOnUse">
          <stop stop-color="#1a1a40"/>
          <stop offset="1" stop-color="#ff3cac"/>
        </linearGradient>
        <radialGradient id="paint1_radial" cx="0" cy="0" r="1" gradientTransform="translate(960 75) scale(960 75)" gradientUnits="userSpaceOnUse">
          <stop stop-color="#00f2fe" stop-opacity="0.5"/>
          <stop offset="1" stop-color="#ff3cac" stop-opacity="0.2"/>
        </radialGradient>
      </defs>
    </svg>
    ''',
    unsafe_allow_html=True
)

# Sayfa başlığı ve açıklama
st.title("🎙️ Konuşma Tanıma Uygulaması")
st.markdown("""
    <div style='padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; backdrop-filter: blur(5px);'>
        <p style='color: white; font-size: 1.2em; margin: 0;'>
            Bu uygulama ses kaydınızı yazıya dönüştürür. Mikrofon kullanarak veya ses dosyası yükleyerek kullanabilirsiniz.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    """Model ve işlemciyi yükle"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model

# Model ve işlemciyi yükle
processor, model = load_model()

# Ses kaydı için parametreler
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 10  # saniye

def record_audio():
    """Mikrofon ile ses kaydet"""
    st.markdown("""
        <div style='padding: 10px; background: rgba(0, 242, 254, 0.1); border-radius: 10px; text-align: center;'>
            <p style='color: white; font-size: 1.2em; margin: 0;'>🎤 Kayıt yapılıyor...</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    audio_data = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32
    )
    sd.wait()
    return audio_data

def process_audio(audio_data, sample_rate):
    """Ses verisini işle ve metne dönüştür"""
    # Ses verisini işle
    input_features = processor(
        audio_data, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).input_features

    if torch.cuda.is_available():
        input_features = input_features.to("cuda")

    # Metne dönüştür
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Kullanıcı arayüzü
st.sidebar.header("⚙️ Ayarlar")
input_method = st.sidebar.radio(
    "Ses Girişi Yöntemi",
    ["🎤 Mikrofon", "📁 Dosya Yükle"]
)

if "🎤 Mikrofon" in input_method:
    if st.button("🎤 Kayda Başla", key="record_button"):
        with st.spinner("🎵 Kayıt yapılıyor..."):
            audio_data = record_audio()
            st.success("✨ Kayıt tamamlandı!")
            
            # Kaydı geçici dosyaya kaydet
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_data, SAMPLE_RATE)
                st.audio(temp_audio.name)
            
            # Ses dosyasını metne dönüştür
            with st.spinner("🔄 Ses metne dönüştürülüyor..."):
                text = process_audio(audio_data.flatten(), SAMPLE_RATE)
                st.markdown("""
                    <div style='padding: 20px; background: rgba(255, 60, 172, 0.1); border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: white; margin: 0;'>📝 Metin:</h3>
                        <p style='color: white; font-size: 1.2em; margin: 10px 0 0 0;'>{}</p>
                    </div>
                    """.format(text), 
                    unsafe_allow_html=True
                )

else:
    uploaded_file = st.file_uploader("📁 Ses dosyası yükle", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        # Dosyayı oku
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)
        
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(suffix="." + uploaded_file.name.split(".")[-1], delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            # Ses dosyasını oku
            audio_data, sample_rate = sf.read(temp_audio.name)
            
            # Ses dosyasını metne dönüştür
            with st.spinner("🔄 Ses metne dönüştürülüyor..."):
                text = process_audio(audio_data, sample_rate)
                st.markdown("""
                    <div style='padding: 20px; background: rgba(255, 60, 172, 0.1); border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: white; margin: 0;'>📝 Metin:</h3>
                        <p style='color: white; font-size: 1.2em; margin: 10px 0 0 0;'>{}</p>
                    </div>
                    """.format(text), 
                    unsafe_allow_html=True
                )

# Uygulama hakkında bilgi
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 10px;'>
    <h3 style='color: white; margin: 0;'>ℹ️ Hakkında</h3>
    <p style='color: white; margin-top: 10px;'>
        Bu uygulama OpenAI'nin Whisper modelini kullanarak ses tanıma yapar.
        <ul style='color: white;'>
            <li>🌍 Türkçe dahil birçok dili destekler</li>
            <li>🎤 Hem mikrofon kaydı hem de dosya yükleme desteklenir</li>
            <li>🤖 Yapay zeka tabanlı ses tanıma</li>
        </ul>
    </p>
</div>
""", unsafe_allow_html=True)