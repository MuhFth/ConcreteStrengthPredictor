# 🏗️ Concrete Strength Predictor - Streamlit Application

Aplikasi prediksi kekuatan beton menggunakan Machine Learning dengan interface Streamlit yang interaktif dan user-friendly.

## ✨ Fitur Utama

### 1. 📝 Input Manual
- Input 8 parameter beton dasar
- Tooltip informatif untuk setiap parameter
- Tombol "Isi Contoh Data" untuk demo cepat
- Hasil prediksi dengan visualisasi komposisi material
- Klasifikasi grade beton otomatis (K-175, K-250, K-300, K-400+)
- Perhitungan W/C ratio otomatis
- Model otomatis menghitung 3 fitur turunan untuk prediksi akurat

### 2. 📁 Upload CSV
- Upload file CSV untuk prediksi batch
- Download template CSV
- Preview data sebelum prediksi
- Statistik hasil prediksi (mean, max, min, std)
- Visualisasi interaktif:
  - Distribusi kekuatan beton
  - Kekuatan vs Umur
  - Distribusi grade
- Download hasil prediksi dalam format CSV

### 3. 📊 Model Performance
- Metrik performa model dari data training:
  - **R² Score: 0.8259** (82.59% variance explained)
  - **RMSE: 7.21 MPa** (Root Mean Squared Error)
  - **MAE: 5.54 MPa** (Mean Absolute Error)
- Feature importance visualization (8 parameter input)
- Riwayat prediksi dengan trend chart
- Model insights dan limitations

## 🚀 Cara Menjalankan

### Prerequisites
Pastikan Python 3.8+ sudah terinstall di sistem Anda.

### Instalasi

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup file model:**

⚠️ **PENTING:** File model harus berada di lokasi yang benar!

**Opsi A (Recommended):** Copy model ke direktori aplikasi
```bash
# Copy model ke direktori yang sama dengan app.py
cp /path/to/concrete_strength_model.pkl .
```

**Opsi B:** Gunakan helper script
```bash
python setup_helper.py
```

**Opsi C:** Update path manual di `app.py` (line ~27):
```python
model_path = 'concrete_strength_model.pkl'  # atau path lengkap Anda
```

3. **Jalankan aplikasi:**
```bash
streamlit run app.py
```

4. **Buka browser:**
Aplikasi akan otomatis terbuka di `http://localhost:8501`

## 📋 Format CSV untuk Upload

File CSV harus memiliki 8 kolom berikut (urutan bebas):

```
Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
```

**Contoh:**
```csv
Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate,Age
540,0,0,162,2.5,1040,676,28
450,100,0,180,3.0,950,720,28
425,106,0,153,4.0,1000,750,28
```

## 📊 Parameter Input

| Parameter | Satuan | Range Tipikal | Deskripsi |
|-----------|--------|---------------|-----------|
| Cement | kg/m³ | 100-540 | Jumlah semen dalam campuran |
| Blast Furnace Slag | kg/m³ | 0-360 | Material pengganti semen |
| Fly Ash | kg/m³ | 0-200 | Abu terbang dari pembakaran batu bara |
| Water | kg/m³ | 120-250 | Jumlah air dalam campuran |
| Superplasticizer | kg/m³ | 0-32 | Bahan kimia untuk workability |
| Coarse Aggregate | kg/m³ | 800-1150 | Agregat kasar (kerikil) |
| Fine Aggregate | kg/m³ | 600-1000 | Agregat halus (pasir) |
| Age | Days | 1-365 | Umur beton dalam hari |

## 🔧 Feature Engineering (Otomatis)

Model secara otomatis menghitung 3 fitur turunan dari 8 input dasar:

1. **Rasio Air-Semen** = Water / Cement
   - Metrik penting untuk kualitas beton
   - Nilai rendah = kekuatan lebih tinggi

2. **Total Bahan Pengikat** = Cement + Slag + Fly Ash
   - Total material yang mengikat agregat

3. **Log Umur** = log(1 + Age)
   - Transformasi untuk menangkap pertumbuhan kekuatan non-linear

### ℹ️ Important Note: Column Name Mapping

Model Anda menggunakan nama kolom dalam **Bahasa Indonesia**. Aplikasi secara otomatis melakukan mapping:
- `Cement` → `Semen`
- `Water` → `Air`
- `Blast Furnace Slag` → `Slag_Tanur_Tinggi`
- `Fly Ash` → `Abu_Terbang`
- `Coarse Aggregate` → `Agregat_Kasar`
- `Fine Aggregate` → `Agregat_Halus`
- `Age` → `Umur_Hari`

Anda **tidak perlu** melakukan mapping manual - aplikasi menangani ini secara otomatis! ✨

📖 **Untuk detail teknis lengkap**, lihat `TECHNICAL_NOTES.md`

## 🎯 Grade Beton

| Grade | Kekuatan (MPa) | Kategori | Penggunaan |
|-------|----------------|----------|------------|
| K-175 | < 20 | 🔴 Rendah | Pekerjaan non-struktural |
| K-250 | 20-30 | 🟡 Sedang | Struktur ringan |
| K-300 | 30-40 | 🟢 Tinggi | Struktur bangunan |
| K-400+ | > 40 | 🔵 Sangat Tinggi | Struktur khusus |

## 🤖 Informasi Model

- **Algorithm:** Linear Regression with Feature Engineering
- **Input Features:** 8 parameters (user input)
- **Model Features:** 11 (8 input + 3 engineered)
- **Output:** Compressive Strength (MPa)
- **Training Performance:**
  - R² Score: 0.8259
  - RMSE: 7.21 MPa
  - MAE: 5.54 MPa

### Fitur Model yang Digunakan:
1. Semen
2. Slag Tanur Tinggi
3. Abu Terbang
4. Air
5. Superplasticizer
6. Agregat Kasar
7. Agregat Halus
8. Umur (Hari)
9. **Rasio Air-Semen** (calculated)
10. **Total Bahan Pengikat** (calculated)
11. **Log Umur** (calculated)

## 💡 Tips Penggunaan

1. **Input Manual:** Gunakan untuk prediksi cepat dan melihat detail komposisi material
2. **Upload CSV:** Ideal untuk prediksi batch dan analisis komparatif
3. **Model Performance:** Pahami bagaimana model bekerja dan faktor-faktor yang mempengaruhi prediksi
4. **W/C Ratio:** Perhatikan rasio air-semen - rasio rendah (0.4-0.5) umumnya menghasilkan beton lebih kuat
5. **Age:** Kekuatan beton meningkat seiring waktu, umumnya mencapai 70% kekuatan pada 7 hari dan 90%+ pada 28 hari

## ⚠️ Catatan Penting

- Model ini menggunakan Linear Regression dengan feature engineering untuk akurasi lebih baik
- Prediksi bergantung pada kualitas input dan berada dalam range data training
- Untuk aplikasi kritis (struktur vital), selalu verifikasi dengan pengujian laboratorium
- Model ter-training dengan data beton normal - hasil untuk beton spesial mungkin kurang akurat
- Pastikan input berada dalam range yang wajar untuk hasil terbaik

## 🛠️ Teknologi yang Digunakan

- **Streamlit:** Framework untuk web app
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing
- **Plotly:** Interactive visualizations
- **Scikit-learn:** Machine learning model
- **Joblib:** Model serialization

## 📞 Support

Jika ada pertanyaan atau masalah:
1. Check error message di interface Streamlit
2. Pastikan model file di lokasi yang benar
3. Verifikasi semua dependencies terinstall
4. Cek Python version (3.8+ required)
5. Lihat SETUP.md untuk panduan detail

## 🎨 Customization

### Ubah Skema Warna
Edit CSS di `app.py` (line ~15-70)

### Update Model Path
Edit fungsi `load_model()` di `app.py` (line ~27)

### Modifikasi Fitur
Edit section input di sekitar line 300+ untuk menyesuaikan parameter

---

**© 2026 Concrete Strength Predictor | Built with ❤️ using Streamlit & Python**
