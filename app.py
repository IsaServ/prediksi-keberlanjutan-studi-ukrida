# ============================================================
#  APP.PY — PREDIKSI KEBERLANJUTAN STUDI MAHASISWA
#  Program Studi Sistem Informasi UKRIDA
#  Model: Random Forest (random_forest_ukrida.sav)
# ============================================================

import io
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Keberlanjutan Studi Mahasiswa — UKRIDA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. KONSTANTA — FITUR, LABEL, DAN ENCODING
# ============================================================
FITUR_MODEL = [
    "Semester_Saat_Ini",
    "IPK_Saat_Ini",
    "IPS_Semester_Berjalan",
    "Total_SKS_Lulus",
    "Status_Keuangan",
    "Total_Poin_Soft_Skill",
    "Bimbingan_Pending",
    "Jumlah_Bimbingan_Semester_Berjalan",
]

LABEL_FITUR = {
    "Semester_Saat_Ini": "Semester Saat Ini",
    "IPK_Saat_Ini": "IPK Saat Ini",
    "IPS_Semester_Berjalan": "IPS Semester Berjalan",
    "Total_SKS_Lulus": "Total SKS Lulus",
    "Status_Keuangan": "Status Keuangan",
    "Total_Poin_Soft_Skill": "Total Poin Soft Skill",
    "Bimbingan_Pending": "Bimbingan Pending",
    "Jumlah_Bimbingan_Semester_Berjalan": "Jumlah Bimbingan Semester Berjalan",
}

# Mapping encoding Status_Keuangan
ENCODING_KEUANGAN = {"Lunas": 1, "Belum Lunas": 0}

NAMA_FILE_MODEL = "random_forest_ukrida.sav"

# Kolom numerik yang wajib divalidasi sebagai angka pada CSV
KOLOM_NUMERIK = [c for c in FITUR_MODEL if c != "Status_Keuangan"]


# ============================================================
# 3. CSS — TEMA WARNA UKRIDA
# ============================================================
st.markdown(
    """
    <style>
        /* ---------- Palet warna ---------- */
        :root {
            --ukrida-navy:       #0F2A52;
            --ukrida-navy-soft:  #1E4577;
            --ukrida-bg:         #F4F6F9;
            --ukrida-card:       #FFFFFF;
            --ukrida-border:     #E2E6EC;
            --ukrida-text:       #1F2937;
            --ukrida-muted:      #6B7280;
            --safe-bg:           #E8F7EE;
            --safe-text:         #15803D;
            --safe-border:       #86E0AA;
            --risk-bg:           #FDECEC;
            --risk-text:         #B91C1C;
            --risk-border:       #F2A6A6;
        }

        /* ---------- Background utama ---------- */
        .stApp {
            background-color: var(--ukrida-bg);
        }

        /* ---------- Sembunyikan elemen default Streamlit ---------- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}

        /* ---------- Header / Banner aplikasi ---------- */
        .ukrida-header {
            background: linear-gradient(135deg, var(--ukrida-navy) 0%, var(--ukrida-navy-soft) 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 1.6rem;
        }
        .ukrida-header .badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.25rem 0.8rem;
            border-radius: 20px;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.9rem;
        }
        .ukrida-header h1 {
            font-size: 2.0rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #FFFFFF;
        }
        .ukrida-header p {
            font-size: 1.0rem;
            color: #DCE6F2;
            margin: 0;
        }

        /* ---------- Section card ---------- */
        .ukrida-section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ukrida-navy);
            margin-bottom: 0.1rem;
        }
        .ukrida-section-sub {
            font-size: 0.88rem;
            color: var(--ukrida-muted);
            margin-bottom: 1rem;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF;
            border-radius: 8px 8px 0 0;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            color: var(--ukrida-muted);
        }
        .stTabs [aria-selected="true"] {
            color: var(--ukrida-navy) !important;
            border-bottom: 3px solid var(--ukrida-navy) !important;
        }

        /* ---------- Tombol utama ---------- */
        .stButton > button {
            background-color: var(--ukrida-navy);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            transition: background-color 0.15s ease-in-out;
        }
        .stButton > button:hover {
            background-color: var(--ukrida-navy-soft);
            color: #FFFFFF;
        }
        .stDownloadButton > button {
            background-color: #FFFFFF;
            color: var(--ukrida-navy);
            border: 1.5px solid var(--ukrida-navy);
            border-radius: 8px;
            font-weight: 600;
        }
        .stDownloadButton > button:hover {
            background-color: var(--ukrida-bg);
        }

        /* ---------- Kotak hasil prediksi ---------- */
        .result-box {
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.8rem;
            border: 1.5px solid;
        }
        .result-box.safe {
            background-color: var(--safe-bg);
            border-color: var(--safe-border);
        }
        .result-box.risk {
            background-color: var(--risk-bg);
            border-color: var(--risk-border);
        }
        .result-title.safe { color: var(--safe-text); font-size: 1.25rem; font-weight: 800; }
        .result-title.risk { color: var(--risk-text); font-size: 1.25rem; font-weight: 800; }
        .result-sub { color: var(--ukrida-muted); font-size: 0.9rem; margin-top: 0.25rem; }

        /* ---------- Footer ---------- */
        .ukrida-footer {
            text-align: center;
            color: var(--ukrida-muted);
            font-size: 0.82rem;
            padding: 1.5rem 0 0.5rem 0;
            border-top: 1px solid var(--ukrida-border);
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD MODEL (.sav) MENGGUNAKAN PICKLE
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Memuat model Random Forest dari file pickle (.sav)."""
    with open(path, "rb") as f:
        return pickle.load(f)


model = None
model_error = None
try:
    model = load_model(NAMA_FILE_MODEL)
except FileNotFoundError:
    model_error = (
        f"File model '{NAMA_FILE_MODEL}' tidak ditemukan. "
        "Pastikan file model berada pada folder yang sama dengan app.py."
    )
except Exception as e:  # noqa: BLE001
    model_error = f"Gagal memuat model: {e}"


# ============================================================
# 5. FUNGSI BANTUAN
# ============================================================
def encode_status_keuangan(nilai: str):
    """Mengonversi 'Lunas'/'Belum Lunas' menjadi 1/0."""
    return ENCODING_KEUANGAN.get(nilai, np.nan)


def get_top3_feature_importance(rf_model):
    """Mengambil 3 fitur dengan importance tertinggi dari model."""
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame(
        {"Fitur": FITUR_MODEL, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df["Label"] = fi_df["Fitur"].map(LABEL_FITUR)
    fi_df["Persentase"] = fi_df["Importance"] / fi_df["Importance"].sum() * 100
    return fi_df.head(3)


def validasi_input_manual(values: dict):
    """Validasi nilai input manual. Mengembalikan (is_valid, error_msg, data_bersih)."""
    errors = []
    data = {}

    for kol in KOLOM_NUMERIK:
        raw = values.get(kol, "").strip()
        if raw == "":
            errors.append(f"Kolom '{LABEL_FITUR[kol]}' belum diisi.")
            continue
        try:
            data[kol] = float(raw)
        except ValueError:
            errors.append(
                f"Nilai pada '{LABEL_FITUR[kol]}' harus berupa angka. "
                f"Nilai yang dimasukkan: '{raw}'."
            )

    status_keu = values.get("Status_Keuangan")
    if status_keu not in ENCODING_KEUANGAN:
        errors.append("Status Keuangan wajib dipilih.")
    else:
        data["Status_Keuangan"] = encode_status_keuangan(status_keu)

    if errors:
        return False, errors, None
    return True, [], data


def validasi_csv(df: pd.DataFrame):
    """
    Validasi dataframe CSV yang diupload.
    Mengembalikan (is_valid, list_error, dataframe_bersih_atau_None)
    """
    errors = []

    # 1) Cek kolom wajib ada
    kolom_hilang = [c for c in FITUR_MODEL if c not in df.columns]
    if kolom_hilang:
        errors.append(
            "Kolom berikut tidak ditemukan pada file CSV: "
            + ", ".join(kolom_hilang)
            + ". Pastikan nama kolom sesuai dengan template."
        )
        return False, errors, None  # tidak bisa lanjut tanpa kolom wajib

    df_valid = df[FITUR_MODEL].copy()

    # 2) Cek nilai numerik valid
    for kol in KOLOM_NUMERIK:
        converted = pd.to_numeric(df_valid[kol], errors="coerce")
        n_invalid = converted.isna().sum() - df_valid[kol].isna().sum()
        if n_invalid > 0:
            errors.append(
                f"Kolom '{LABEL_FITUR[kol]}' memiliki {int(n_invalid)} "
                f"baris dengan nilai bukan angka yang valid."
            )
        df_valid[kol] = converted

    # 3) Cek missing value pada seluruh kolom wajib
    n_missing_total = df_valid[KOLOM_NUMERIK].isna().sum().sum()
    if n_missing_total > 0 and not errors:
        errors.append(
            f"Ditemukan {int(n_missing_total)} nilai kosong pada kolom numerik."
        )

    # 4) Cek nilai Status_Keuangan valid
    nilai_tidak_valid = (
        ~df_valid["Status_Keuangan"].isin(ENCODING_KEUANGAN.keys())
    ).sum()
    if nilai_tidak_valid > 0:
        errors.append(
            f"Kolom 'Status_Keuangan' memiliki {int(nilai_tidak_valid)} "
            "baris dengan nilai selain 'Lunas' atau 'Belum Lunas'."
        )

    if errors:
        return False, errors, None

    # Encoding Status_Keuangan untuk keperluan prediksi
    df_valid["Status_Keuangan_Encoded"] = df_valid["Status_Keuangan"].map(
        ENCODING_KEUANGAN
    )

    return True, [], df_valid


def buat_template_csv() -> str:
    """Membuat 1 baris contoh data untuk template CSV."""
    contoh = pd.DataFrame(
        [
            {
                "Semester_Saat_Ini": 4,
                "IPK_Saat_Ini": 3.73,
                "IPS_Semester_Berjalan": 3.55,
                "Total_SKS_Lulus": 72,
                "Status_Keuangan": "Lunas",
                "Total_Poin_Soft_Skill": 85,
                "Bimbingan_Pending": 1,
                "Jumlah_Bimbingan_Semester_Berjalan": 3,
            }
        ]
    )
    return contoh.to_csv(index=False)


# ============================================================
# 6. HEADER APLIKASI
# ============================================================
st.markdown(
    """
    <div class="ukrida-header">
        <div class="badge">PROGRAM STUDI SISTEM INFORMASI</div>
        <h1>🎓 Prediksi Keberlanjutan Studi Mahasiswa</h1>
        <p>Sistem Pendukung Keputusan Akademik — Universitas Kristen Krida Wacana (UKRIDA)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(model_error)
    st.stop()


# ============================================================
# 7. TAB NAVIGASI
# ============================================================
tab_manual, tab_csv = st.tabs(["📋  Input Data Manual", "📤  Prediksi Massal (CSV)"])


# ════════════════════════════════════════════════════════════
# TAB 1 — INPUT DATA MANUAL
# ════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown(
        '<div class="ukrida-section-title">Input Data Mahasiswa</div>'
        '<div class="ukrida-section-sub">'
        "Isi data akademik mahasiswa untuk mendapatkan prediksi individu."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
        
            in_semester = st.text_input(
                "Semester Saat Ini",
                placeholder="Contoh: 4"
            )
        
            in_ipk = st.text_input(
                "IPK Saat Ini",
                placeholder="Contoh: 3.73"
            )
        
            in_ips = st.text_input(
                "IPS Semester Berjalan",
                placeholder="Contoh: 3.55"
            )
        
            in_sks = st.text_input(
                "Total SKS Lulus",
                placeholder="Contoh: 72"
            )
        
        with col2:
        
            in_keuangan = st.selectbox(
                "Status Keuangan",
                ["Lunas", "Belum Lunas"]
            )
        
            in_softskill = st.text_input(
                "Total Poin Soft Skill",
                placeholder="Contoh: 85"
            )
        
            in_bimb_pending = st.text_input(
                "Bimbingan Pending",
                placeholder="Contoh: 1"
            )
        
            in_bimb_jumlah = st.text_input(
                "Jumlah Bimbingan Semester Berjalan",
                placeholder="Contoh: 3"
            )

        st.caption("＊ Semua field wajib diisi untuk mendapatkan hasil prediksi yang akurat.")
        klik_prediksi = st.button("📊  Prediksi Mahasiswa", use_container_width=False)

    if klik_prediksi:
        raw_values = {
            "Semester_Saat_Ini": in_semester,
            "IPK_Saat_Ini": in_ipk,
            "IPS_Semester_Berjalan": in_ips,
            "Total_SKS_Lulus": in_sks,
            "Status_Keuangan": in_keuangan,
            "Total_Poin_Soft_Skill": in_softskill,
            "Bimbingan_Pending": in_bimb_pending,
            "Jumlah_Bimbingan_Semester_Berjalan": in_bimb_jumlah,
        }

        is_valid, errors, data_bersih = validasi_input_manual(raw_values)

        if not is_valid:
            st.error("Mohon perbaiki input berikut sebelum melanjutkan:")
            for e in errors:
                st.markdown(f"- {e}")
        else:
            # Susun fitur sesuai urutan training, gunakan DataFrame
            # agar nama kolom tervalidasi oleh scikit-learn.
            x_input = pd.DataFrame([data_bersih])[FITUR_MODEL]

            pred_class = model.predict(x_input)[0]
            pred_proba = model.predict_proba(x_input)[0]
            risk_proba = pred_proba[1] * 100  # probabilitas kelas 1 (tidak melanjutkan)

            st.markdown("### Hasil Prediksi")

            if pred_class == 0:
                st.markdown(
                    f"""
                    <div class="result-box safe">
                        <div class="result-title safe">✅ Aman Melanjutkan Studi</div>
                        <div class="result-sub">Probabilitas Risiko: {risk_proba:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box risk">
                        <div class="result-title risk">⚠️ Berisiko Tidak Melanjutkan Studi</div>
                        <div class="result-sub">Probabilitas Risiko: {risk_proba:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---------- Top 3 Faktor Paling Berpengaruh ----------
            st.markdown("#### Faktor Paling Berpengaruh")
            st.caption(
                "Berdasarkan feature importance global dari model Random Forest."
            )
            top3 = get_top3_feature_importance(model)
            for i, row in top3.iterrows():
                st.markdown(f"**{i + 1}. {row['Label']}** — {row['Persentase']:.1f}%")
                st.progress(min(float(row["Persentase"]) / 100, 1.0))


# ════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI MASSAL MENGGUNAKAN CSV
# ════════════════════════════════════════════════════════════
with tab_csv:
    st.markdown(
        '<div class="ukrida-section-title">Prediksi Massal Menggunakan File CSV</div>'
        '<div class="ukrida-section-sub">'
        "Unggah file CSV untuk memproses banyak mahasiswa sekaligus."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Drag & drop CSV di sini, atau klik untuk memilih file",
            type=["csv"],
            help="Hanya menerima file berformat .csv",
        )

        st.markdown("---")
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(
                "**Template CSV**  \n"
                "Unduh template untuk memastikan format kolom yang benar."
            )
        with col_t2:
            st.download_button(
                label="⬇️  Download Template",
                data=buat_template_csv(),
                file_name="template_prediksi_mahasiswa.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
            
                # Bersihkan kolom Unnamed
                df_upload = df_upload.loc[
                    :, ~df_upload.columns.str.contains("^Unnamed")
                ]
            
                # Bersihkan kolom kosong
                df_upload = df_upload.dropna(
                    axis=1,
                    how="all"
                )
            
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
                df_upload = None

            if df_upload is not None:
                st.markdown("**Pratinjau Data:**")
                st.dataframe(
                    df_upload[FITUR_MODEL].head(5),
                    use_container_width=True
                )

                proses_csv = st.button("📊  Proses File CSV")

                if proses_csv:
                    is_valid, errors, df_bersih = validasi_csv(df_upload)

                    if not is_valid:
                        st.error(
                            "File CSV tidak dapat diproses karena terdapat "
                            "kesalahan berikut:"
                        )
                        for e in errors:
                            st.markdown(f"- {e}")
                    else:
                        # ---------- Prediksi Massal ----------
                        x_batch = df_bersih[FITUR_MODEL].copy()
                        x_batch["Status_Keuangan"] = (
                            df_bersih["Status_Keuangan_Encoded"]
                        )

                        pred_classes = model.predict(x_batch)
                        pred_probas = model.predict_proba(x_batch)[:, 1] * 100

                        hasil_df = df_upload.loc[df_bersih.index].copy()
                        hasil_df["Prediksi"] = np.where(
                            pred_classes == 0,
                            "Aman Melanjutkan Studi",
                            "Berisiko Tidak Melanjutkan Studi",
                        )
                        hasil_df["Probabilitas_Risiko"] = np.round(pred_probas, 1)

                        top3 = get_top3_feature_importance(model)

                        hasil_df["Faktor_1"] = top3.iloc[0]["Label"]
                        hasil_df["Persentase_1"] = round(top3.iloc[0]["Persentase"], 1)
                        
                        hasil_df["Faktor_2"] = top3.iloc[1]["Label"]
                        hasil_df["Persentase_2"] = round(top3.iloc[1]["Persentase"], 1)
                        
                        hasil_df["Faktor_3"] = top3.iloc[2]["Label"]
                        hasil_df["Persentase_3"] = round(top3.iloc[2]["Persentase"], 1)

                        n_total = len(hasil_df)
                        n_risk = int((pred_classes == 1).sum())
                        n_safe = int((pred_classes == 0).sum())

                        st.markdown("### Ringkasan Hasil Prediksi")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Mahasiswa Diproses", f"{n_total}")
                        m2.metric("Berisiko", f"{n_risk}")
                        m3.metric("Aman", f"{n_safe}")

                        st.markdown("**Detail Hasil Prediksi:**")
                        st.dataframe(hasil_df, use_container_width=True)

                        csv_buffer = io.StringIO()
                        hasil_df.to_csv(csv_buffer, index=False)

                        st.download_button(
                            label="⬇️  Download Hasil Prediksi",
                            data=csv_buffer.getvalue(),
                            file_name="hasil_prediksi_mahasiswa.csv",
                            mime="text/csv",
                        )


# ============================================================
# 8. FOOTER
# ============================================================
st.markdown(
    """
    <div class="ukrida-footer">
        UKRIDA · Program Studi Sistem Informasi · Sistem Prediksi Keberlanjutan Studi
    </div>
    """,
    unsafe_allow_html=True,
)
