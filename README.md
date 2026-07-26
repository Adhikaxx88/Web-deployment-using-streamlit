

```markdown
# Student Placement Predictor

Aplikasi web berbasis Streamlit untuk memprediksi status penempatan kerja (placement) dan mengestimasi gaji (salary) mahasiswa, menggunakan model machine learning yang sudah dilatih sebelumnya.

## Fitur

Aplikasi ini memiliki dua mode prediksi yang bisa dipilih lewat sidebar:

1. **Classification** — Memprediksi apakah seorang mahasiswa akan Placed atau Not Placed, lengkap dengan probabilitas, gauge chart, dan bar chart perbandingan.
2. **Regression** — Mengestimasi salary (LPA) mahasiswa yang diasumsikan sudah Placed, dibandingkan dengan benchmark rata-rata (min, rata-rata, dan maksimum).

Input data mahasiswa mencakup tiga kategori:
- **Akademik**: CGPA, jumlah backlog, internship, sertifikasi
- **Skill & Aktivitas**: rating coding, komunikasi, aptitude, keterlibatan ekstrakurikuler
- **Gaya Hidup & Demografi**: jam belajar/tidur, tingkat stres, gender, jurusan, tingkat pendapatan keluarga, tier kota, akses internet, kerja paruh waktu

## Model

**Classification**
- Model: MLP Classifier
- Target: `placement_status`
- Metrik: Accuracy 89%, AUC 0.899

**Regression**
- Model: Linear Regression
- Target: `salary_lpa`
- Catatan: Dilatih hanya pada data mahasiswa yang Placed

Model disimpan dalam format `.pkl`:
- `best_model_classification.pkl`
- `best_model_regression.pkl`
## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Plotly (visualisasi gauge chart & bar chart)

## Struktur Project

```
├── app.py                          # Main Streamlit app
├── best_model_classification.pkl   # Model klasifikasi (MLP)
├── best_model_regression.pkl       # Model regresi (Linear Regression)
└── requirements.txt                # Dependencies
```

## Cara Menjalankan

1. Clone repository ini:
   ```bash
   git clone https://github.com/Adhikaxx88/Web-deployment-using-streamlit.git
   cd Web-deployment-using-streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Buka browser di `http://localhost:8501`

## Cara Pakai

1. Pilih mode prediksi di sidebar (Classification atau Regression)
2. Isi data mahasiswa pada form yang tersedia
3. Klik tombol Prediksi/Estimasi Salary
4. Lihat hasil prediksi beserta visualisasi probabilitas/estimasinya
```
