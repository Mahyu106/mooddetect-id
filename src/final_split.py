# src/final_split.py
# Membagi data: 90% untuk melatih model final, 10% hold-out
# (untuk memilih epoch terbaik dan menentukan ambang "tidak yakin").
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Folder utama proyek (satu tingkat di atas folder src)
akar = Path(__file__).resolve().parents[1]
data = pd.read_csv(akar / "data" / "processed" / "emotweetid_preprocessed.csv")

semua_indeks = list(range(len(data)))

# Pisahkan 10% secara stratified; seed 42 supaya hasilnya selalu sama
indeks_latih, indeks_holdout = train_test_split(
    semua_indeks,
    test_size=0.10,
    stratify=data["label"],
    random_state=42,
)

# Simpan: satu baris per teks, kolom peran = train atau holdout
hasil = pd.DataFrame({"indeks": semua_indeks, "peran": "train"})
hasil.loc[hasil["indeks"].isin(indeks_holdout), "peran"] = "holdout"
hasil.to_csv(akar / "data" / "processed" / "final_split.csv", index=False)

# Ringkasan untuk dicek
print("Jumlah train  :", (hasil["peran"] == "train").sum())
print("Jumlah holdout:", (hasil["peran"] == "holdout").sum())
print()
print("Sebaran emosi di holdout:")
print(data.loc[indeks_holdout, "label"].value_counts().sort_index())