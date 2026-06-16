# Prediksi Keberlanjutan Studi Mahasiswa — UKRIDA

Aplikasi Streamlit untuk memprediksi risiko keberlanjutan studi mahasiswa
Program Studi Sistem Informasi UKRIDA menggunakan model Random Forest.

---

## 📁 Struktur Folder Project

```
ukrida-streamlit-app/
├── app.py                              # Aplikasi utama Streamlit
├── random_forest_ukrida.sav            # Model Random Forest
├── requirements.txt                    # Daftar dependency Python
├── template_prediksi_mahasiswa.csv     # Template contoh untuk upload massal
└── README.md                           # Dokumentasi ini
```

---

## 🚀 Menjalankan Aplikasi Secara Lokal

1. Pastikan Python 3.9 atau lebih baru sudah terinstal.
2. Install seluruh dependency:

   ```bash
   pip install -r requirements.txt
   ```

3. Letakkan file `random_forest_ukrida.sav` pada folder yang sama dengan `app.py`.
4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

5. Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## 📤 Mengunggah Project ke GitHub

1. Buat repository baru di GitHub (misalnya `prediksi-keberlanjutan-studi-ukrida`).
2. Inisialisasi git pada folder project:

   ```bash
   cd ukrida-streamlit-app
   git init
   git add app.py requirements.txt template_prediksi_mahasiswa.csv README.md
   git add random_forest_ukrida.sav
   git commit -m "Initial commit: Streamlit app prediksi keberlanjutan studi UKRIDA"
   ```

3. Hubungkan ke repository GitHub dan push:

   ```bash
   git remote add origin https://github.com/USERNAME/NAMA-REPO.git
   git branch -M main
   git push -u origin main
   ```

> **Catatan ukuran file:** Jika file `.sav` berukuran besar (>25 MB), GitHub
> akan menampilkan peringatan. Untuk file di bawah 100 MB, hal ini masih
> bisa diunggah secara normal namun disarankan menggunakan Git LFS jika
> ukurannya mendekati batas tersebut.

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

1. Pastikan repository GitHub Anda bersifat **public** (atau Anda memiliki
   akses Streamlit Cloud untuk repository private).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login menggunakan
   akun GitHub Anda.
3. Klik **"New app"**.
4. Pilih:
   - **Repository**: repository yang sudah Anda buat sebelumnya
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **"Deploy"**.
6. Tunggu proses build selesai (Streamlit Cloud akan otomatis menginstal
   dependency dari `requirements.txt`).
7. Setelah selesai, aplikasi akan memiliki URL publik seperti:
   `https://nama-app-anda.streamlit.app`

### Hal yang Perlu Diperhatikan Saat Deploy

- **Versi scikit-learn**: Pastikan versi `scikit-learn` pada `requirements.txt`
  kompatibel dengan versi yang digunakan saat melatih model di Google Colab.
  Jika terjadi `InconsistentVersionWarning` saat load model, periksa versi
  scikit-learn di Colab dengan menjalankan:

  ```python
  import sklearn
  print(sklearn.__version__)
  ```

  lalu sesuaikan versi pada `requirements.txt` agar sama atau kompatibel.

- **Ukuran file model**: Streamlit Cloud memiliki batas resource gratis.
  Model Random Forest dengan `n_estimators=100` pada dataset ±400 baris
  umumnya berukuran kecil (di bawah 5 MB) sehingga tidak menjadi masalah.

- **Update aplikasi**: Setiap kali Anda melakukan `git push` ke branch `main`,
  Streamlit Cloud akan otomatis melakukan redeploy aplikasi.

---

## 🧩 Fitur Aplikasi

| Fitur | Deskripsi |
|---|---|
| Input Data Manual | Form input untuk prediksi satu mahasiswa |
| Prediksi Massal (CSV) | Upload file CSV untuk memproses banyak mahasiswa sekaligus |
| Download Template CSV | Template contoh format kolom yang benar |
| Hasil Prediksi Manual | Status (Aman/Berisiko) + probabilitas risiko |
| Top 3 Faktor Berpengaruh | Feature importance dari model Random Forest |
| Download Hasil Prediksi CSV | Hasil prediksi massal dapat diunduh kembali |
| Validasi Input | Validasi nama kolom, kelengkapan data, dan tipe data numerik |

---

## ⚠️ Catatan Penting

- Urutan kolom fitur pada `app.py` (variabel `FITUR_MODEL`) **harus sama persis**
  dengan urutan kolom `X_train` saat model dilatih di Google Colab (BAB IV).
  Jika urutan berbeda, prediksi yang dihasilkan tidak akan valid.
- Encoding `Status_Keuangan` (`Lunas` → 1, `Belum Lunas` → 0) harus konsisten
  dengan encoding yang digunakan pada tahap preprocessing (BAB III).
- Aplikasi ini ditujukan sebagai alat bantu (decision support), bukan
  pengganti keputusan akademik oleh dosen pembimbing maupun pihak Program Studi.
