"""
Fungsi pembersihan teks untuk proyek deteksi emosi.

Ada dua pipeline:
- preprocess_minimal : untuk IndoBERT (tanda baca dan huruf besar/kecil dipertahankan)
- preprocess_penuh   : untuk SVM dan BiLSTM (pembersihan lebih agresif)
"""
import re
import emoji  # library untuk mengubah emoji menjadi kata

# Kamus slang awal (hanya singkatan umum). Ini titik awal, bukan kamus lengkap;
# nanti bisa diperluas setelah kita melihat kata-kata yang sering muncul di data.
KAMUS_SLANG = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "tdk": "tidak", "yg": "yang", "dgn": "dengan",
    "utk": "untuk", "krn": "karena", "karna": "karena", "krna": "karena",
    "bgt": "banget", "bngt": "banget", "sm": "sama", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "blm": "belum", "belom": "belum",
    "tp": "tapi", "jgn": "jangan", "emg": "memang", "emang": "memang",
    "kalo": "kalau", "klo": "kalau", "aja": "saja", "lg": "lagi",
    "dlm": "dalam", "dr": "dari", "pdhl": "padahal", "jg": "juga",
    "org": "orang", "sy": "saya", "skrg": "sekarang", "hrs": "harus",
    "bs": "bisa", "knp": "kenapa", "tmn": "teman", "trs": "terus",
}


def _emoji_ke_kata(teks):
    """Ubah emoji menjadi kata, misalnya 😭 menjadi 'wajah menangis keras'."""
    try:
        hasil = emoji.demojize(teks, language="id", delimiters=(" ", " "))
    except Exception:
        # Cadangan jika bahasa Indonesia tidak didukung versi library ini
        hasil = emoji.demojize(teks, delimiters=(" ", " "))
    return hasil.replace("_", " ")


def _hapus_url_mention_hashtag(teks):
    """Hapus URL dan @mention; untuk #hashtag, hapus simbol # saja dan simpan katanya."""
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)  # hapus URL
    teks = re.sub(r"@\w+", " ", teks)                   # hapus mention
    teks = re.sub(r"#(\w+)", r"\1", teks)               # #senang -> senang
    return teks


def _rapikan_spasi(teks):
    """Ubah spasi berlebihan (juga baris baru dan tab) menjadi satu spasi."""
    return re.sub(r"\s+", " ", teks).strip()


def _ulang_dua(teks):
    """Ubah penulisan pengulangan kata: 'hati2' menjadi 'hati hati'."""
    return re.sub(r"\b([A-Za-z]{2,})2\b", r"\1 \1", teks)


def normalisasi_slang(teks):
    """
    Ganti singkatan/slang dengan bentuk baku memakai KAMUS_SLANG.
    Tidak peka huruf besar/kecil, dan tetap bekerja pada kata yang
    menempel dengan tanda baca (misalnya 'bgt!!').
    """
    def ganti(cocok):
        kata = cocok.group(0)
        return KAMUS_SLANG.get(kata.lower(), kata)
    return re.sub(r"[A-Za-z]+", ganti, teks)


def preprocess_minimal(teks, pakai_slang=False):
    """
    Pipeline minimal untuk IndoBERT.
    Urutan: emoji -> URL/mention/hashtag -> (opsional) pengulangan dan slang
    -> rapikan spasi. Tanda baca dan huruf besar/kecil sengaja TIDAK diubah.
    """
    teks = _emoji_ke_kata(str(teks))
    teks = _hapus_url_mention_hashtag(teks)
    if pakai_slang:
        teks = _ulang_dua(teks)
        teks = normalisasi_slang(teks)
    return _rapikan_spasi(teks)


def preprocess_penuh(teks):
    """
    Pipeline penuh untuk SVM dan BiLSTM.
    Urutan: emoji -> URL/mention/hashtag -> pengulangan 'kata2' -> huruf kecil
    -> hapus angka dan tanda baca -> normalisasi slang -> rapikan spasi.
    """
    teks = _emoji_ke_kata(str(teks))
    teks = _hapus_url_mention_hashtag(teks)
    teks = _ulang_dua(teks)
    teks = teks.lower()
    teks = re.sub(r"[^a-z\s]", " ", teks)  # sisakan hanya huruf a-z dan spasi
    teks = normalisasi_slang(teks)
    return _rapikan_spasi(teks)