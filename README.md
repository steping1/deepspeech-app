# 🎙️ Konuşma Tanıma Uygulaması (DeepSpeech App)

Bu uygulama, OpenAI'nin Whisper modelini kullanarak ses kayıtlarını gerçek zamanlı olarak metne dönüştüren modern bir Streamlit uygulamasıdır.

## ✨ Özellikler

- 🎤 **Canlı Mikrofon Kaydı**: Doğrudan mikrofondan ses kaydı yapabilme
- 📁 **Dosya Yükleme**: WAV, MP3, M4A formatlarında ses dosyalarını yükleme
- 🌍 **Çoklu Dil Desteği**: Türkçe dahil birçok dilde ses tanıma
- 🎨 **Modern Arayüz**: Gradient arka plan ve animasyonlu tasarım
- ⚡ **GPU Desteği**: CUDA destekli hızlı işleme
- 🤖 **AI Tabanlı**: OpenAI Whisper modelini kullanır

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- pip
- (Opsiyonel) CUDA destekli GPU

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/steping1/deepspeech-app.git
cd deepspeech-app
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
pip install streamlit  # Streamlit'i ayrıca yükleyin
```

## 🎯 Kullanım

### Uygulamayı Başlatın
```bash
streamlit run app.py
```

### Tarayıcıda Açın
Uygulama otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### Kullanım Seçenekleri

#### 🎤 Mikrofon ile Kayıt
1. Sol panelden "🎤 Mikrofon" seçeneğini seçin
2. "🎤 Kayda Başla" butonuna tıklayın
3. 10 saniye boyunca konuşun
4. Kayıt otomatik olarak metne dönüştürülecektir

#### 📁 Dosya Yükleme
1. Sol panelden "📁 Dosya Yükle" seçeneğini seçin
2. Desteklenen formatlardan birini (WAV, MP3, M4A) yükleyin
3. Dosya otomatik olarak işlenip metne dönüştürülecektir

## 🧪 Test

Whisper modelinin doğru yüklendiğini test etmek için:
```bash
python test_whisper.py
```

## 📦 Bağımlılıklar

- **torch**: PyTorch framework
- **transformers**: Hugging Face Transformers (Whisper modeli için)
- **sounddevice**: Mikrofon kaydı için
- **soundfile**: Ses dosyası işleme
- **numpy**: Numerik işlemler
- **scipy**: Sinyal işleme
- **streamlit**: Web arayüzü

## 🔧 Konfigürasyon

### Ses Kayıt Parametreleri
- **Örnekleme Oranı**: 16,000 Hz
- **Kanal Sayısı**: 1 (Mono)
- **Kayıt Süresi**: 10 saniye

### Model Ayarları
- **Kullanılan Model**: `openai/whisper-small`
- **Dil Desteği**: Çoklu dil (Türkçe dahil)
- **GPU Desteği**: Otomatik algılama

## 🎨 Arayüz Özellikleri

- **Gradient Arka Plan**: Modern görsel tasarım
- **Responsive Tasarım**: Farklı ekran boyutlarına uyum
- **Animasyonlu Butonlar**: Hover efektleri
- **Dalga Efektleri**: SVG tabanlı görsel öğeler
- **Blur Efektleri**: Cam efekti tasarım

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🔗 Bağlantılar

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!