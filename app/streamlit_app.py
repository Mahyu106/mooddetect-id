# app/streamlit_app.py
# Aplikasi web: ketik teks, lihat emosi yang terdeteksi (dengan consent dan disclaimer).
import sys
from pathlib import Path

import streamlit as st

# Agar "from src.predict import ..." bisa ditemukan saat app dijalankan dari folder mana pun
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import prediksi_emosi

NAMA_EMOSI = {
    "anger": "Marah",
    "disgust": "Jijik / muak",
    "fear": "Takut",
    "joy": "Senang",
    "sadness": "Sedih",
    "surprise": "Terkejut",
}

st.set_page_config(page_title="Deteksi Emosi dari Teks", page_icon="💬")
st.title("Deteksi Emosi dari Teks")
st.write("Ketik teks pendek berbahasa Indonesia, lalu lihat emosi yang terdeteksi.")

# Disclaimer: selalu tampil, tidak bisa disembunyikan
st.info(
    "**Penting:** alat ini hanya memperkirakan emosi yang diungkapkan dalam satu teks. "
    "Ini **bukan diagnosis** dan **bukan pengganti psikolog atau tenaga kesehatan**. "
    "Model bisa salah, terutama pada teks panjang, sarkasme, atau teks yang tidak "
    "mengandung emosi."
)


@st.cache_resource
def siapkan_model():
    """Muat model sekali saja; dipakai ulang di setiap klik tombol."""
    prediksi_emosi("tes")  # memanaskan model (mengunduh dari Hub bila perlu)
    return True


with st.spinner("Memuat model, mohon tunggu..."):
    siapkan_model()

# Consent: hasil hanya muncul setelah dicentang
setuju = st.checkbox(
    "Saya mengerti bahwa ini hanya perkiraan otomatis (bukan diagnosis), dan saya "
    "setuju teks saya diproses oleh model untuk mendapatkan hasil ini."
)
st.caption(
    "Teks kamu hanya diproses di memori untuk menghasilkan prediksi. Aplikasi ini "
    "tidak menyimpan teks ke file atau basis data."
)

teks = st.text_area("Teks kamu:", height=120, max_chars=1000)

if st.button("Deteksi emosi", disabled=not setuju):
    if not teks.strip():
        st.warning("Silakan ketik teks dulu.")
    else:
        hasil = prediksi_emosi(teks)
        if hasil["label"] is None:
            st.warning("Teks tidak bisa diproses. Coba ketik kalimat lain.")
        elif hasil["tidak_yakin"]:
            st.subheader("Tidak yakin")
            st.write(
                "Model kurang yakin dengan emosi dalam teks ini. Bisa jadi teksnya netral, "
                "campuran, atau terlalu ambigu."
            )
            tebakan = NAMA_EMOSI[hasil["label"]]
            st.caption(
                f"Tebakan terdekat (hanya petunjuk, jangan dijadikan patokan): "
                f"{tebakan}, confidence {hasil['confidence'] * 100:.0f}%."
            )
        else:
            st.subheader(f"Emosi terdeteksi: {NAMA_EMOSI[hasil['label']]}")
            st.write(f"Confidence: {hasil['confidence'] * 100:.0f}%")
            st.caption(
                "Confidence model cenderung terlalu percaya diri; angka tinggi "
                "tidak menjamin hasilnya benar."
            )
        with st.expander("Lihat probabilitas semua emosi"):
            st.bar_chart({NAMA_EMOSI[k]: v for k, v in hasil["probabilitas"].items()})

if not setuju:
    st.caption("Centang persetujuan di atas untuk mengaktifkan tombol.")

# Rujukan bantuan kesehatan mental (nomor akan diisi dari sumber resmi)
st.divider()
st.subheader("Butuh bicara dengan seseorang?")
st.write(
    "Jika kamu sedang merasa tertekan atau berpikir untuk menyakiti diri sendiri, "
    "kamu tidak sendirian. Hubungi tenaga profesional atau layanan bantuan."
)
st.write(
    "- **Healing119 (Kemenkes), gratis:** telepon **119 lalu pilih ekstensi 8**.\n"
    "- Atau lewat situs [healing119.id](https://healing119.id) (panggilan suara atau chat).\n"
    "- Jika ada bahaya langsung bagi nyawa, hubungi **119** atau layanan gawat darurat terdekat."
)
st.caption(
    "Layanan ini memberi pertolongan psikologis awal, bukan pengganti terapi. "
    "Nomor dan layanan dapat berubah; periksa situs resminya. Informasi per November 2026."
)