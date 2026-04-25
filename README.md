# 🧠 Introvert vs Extrovert Predictor

Kişilik özelliklerine göre bir kişinin **içe dönük (Introvert)** mi yoksa **dışa dönük (Extrovert)** mi olduğunu tahmin eden makine öğrenmesi uygulaması.

---

## 📊 Proje Hakkında

Bu proje **Kaggle Playground Series S5E7** yarışması için geliştirilmiştir. 18,524 kişilik eğitim verisiyle **~%96 doğruluk** oranı elde edilmiştir.

**Kullanılan Özellikler:**
| Özellik | Açıklama |
|---|---|
| Time_spent_Alone | Günde yalnız geçirilen saat |
| Stage_fear | Sahne korkusu (Yes/No) |
| Social_event_attendance | Sosyal etkinliklere katılım sıklığı |
| Going_outside | Dışarı çıkma sıklığı |
| Drained_after_socializing | Sosyalleşme sonrası enerji kaybı (Yes/No) |
| Friends_circle_size | Arkadaş çevresi büyüklüğü |
| Post_frequency | Sosyal medya gönderi sıklığı |

---

## 🚀 Kurulum ve Kullanım

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Modeli eğit ve kaydet
python save_model.py

# 3. Streamlit uygulamasını başlat
streamlit run app.py
```

---

## 🛠️ Kullanılan Teknikler

- **KNN Imputation** – Eksik değerlerin doldurulması
- **StandardScaler** – Özellik ölçeklendirme
- **CatBoost Classifier** – Gradient boosting modeli

---

## 📁 Proje Yapısı

```
├── train.csv                                        # Eğitim verisi
├── test.csv                                         # Test verisi
├── predict-the-introverts-from-the-extroverts-0-96.ipynb  # Analiz notebook'u
├── save_model.py                                    # Model eğitimi ve kaydetme
├── app.py                                           # Streamlit uygulaması
├── requirements.txt                                 # Bağımlılıklar
├── model.joblib                                     # Eğitilmiş model (save_model.py sonrası)
├── scaler.joblib                                    # Scaler (save_model.py sonrası)
└── feature_columns.joblib                           # Özellik sırası (save_model.py sonrası)
```

---

## 📈 Model Performansı

- **Accuracy:** ~%96
- **Algoritma:** CatBoost Classifier
- **Veri Kaynağı:** [Kaggle Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7)
