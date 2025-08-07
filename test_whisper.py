import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main():
    # GPU varsa kullan, yoksa CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Modeli ve işlemciyi yükle
    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = model.to(device)
    
    # Test mesajı
    print("Whisper model loaded successfully!")
    print("Model is ready for speech recognition.")
    print(f"Model language support: Multilingual")

if __name__ == "__main__":
    main()