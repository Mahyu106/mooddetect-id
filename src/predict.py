"""
Prediksi emosi dari satu teks memakai model final di Hugging Face Hub.
Keluaran: label, confidence, keputusan "tidak yakin", dan probabilitas semua emosi.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import preprocess_minimal

REPO_MODEL = "Light2626/indobert-mood"

_cache = {}  # model dimuat sekali saja, lalu dipakai ulang


def _muat():
    """Muat model, tokenizer, dan konfigurasi dari Hub (hanya sekali)."""
    if "model" not in _cache:
        path_konfig = hf_hub_download(REPO_MODEL, "konfigurasi_aplikasi.json")
        with open(path_konfig, encoding="utf-8") as f:
            konfigurasi = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(REPO_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(REPO_MODEL)
        model.eval()
        _cache.update(model=model, tokenizer=tokenizer, konfigurasi=konfigurasi)
    return _cache["model"], _cache["tokenizer"], _cache["konfigurasi"]


def prediksi_emosi(teks):
    """Prediksi emosi satu teks. Mengembalikan dictionary."""
    model, tokenizer, konfigurasi = _muat()
    label = konfigurasi["urutan_label"]
    ambang = konfigurasi["ambang_tidak_yakin"]

    bersih = preprocess_minimal(teks, pakai_slang=False)
    if not bersih.strip():
        return {"label": None, "confidence": 0.0, "tidak_yakin": True,
                "probabilitas": {}, "alasan": "teks kosong"}

    masukan = tokenizer(bersih, truncation=True, max_length=konfigurasi["max_length"],
                        return_tensors="pt")
    with torch.no_grad():
        logit = model(**masukan).logits
    prob = torch.softmax(logit, dim=1)[0].tolist()

    terbaik = max(range(len(label)), key=lambda i: prob[i])
    return {
        "label": label[terbaik],
        "confidence": prob[terbaik],
        "tidak_yakin": prob[terbaik] < ambang,
        "probabilitas": {label[i]: prob[i] for i in range(len(label))},
        "alasan": "",
    }