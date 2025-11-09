import os, re, glob
import numpy as np
import pandas as pd

# === Einstellungen ===
DATA_GLOB = "labdata/Zusammenfassung_*.txt"   # Pfad zu deinen TXT-Files
OUTPUT_CSV = "pelton_master.csv"            # Output-Datei
KEEP_COLS = ["MP_Name", "eta", "Q_A", "n_rpm", "H_tot", "T_Water", "rho"]  # wichtigste Messgrössen
ABRASION_MAP = "abrasion_mapping.csv"       # optional: Mapping Stage → Abrasion (mm/%)

# === Hilfsfunktion: Einzelnes File einlesen ===
def parse_one_txt(path: str) -> pd.DataFrame:
    # 1. Spaltennamen lesen (erste Zeile), zweite Zeile (Einheiten) überspringen
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        f.readline()

    # 2. Daten einlesen
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, names=header, engine="python")

    # 3. Stage-ID aus Dateinamen extrahieren, z. B. „Zusammenfassung_A1_26_10_22.txt“ → „A1“
    m = re.search(r"Zusammenfassung_([A-Za-z0-9]+)_", os.path.basename(path))
    stage_id = m.group(1) if m else os.path.basename(path)
    df["stage_id"] = stage_id

    # 4. Nur relevante Spalten behalten
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP_COLS + ["stage_id"]].copy()

    # 5. Numerische Spalten casten
    for c in KEEP_COLS:
        if c != "MP_Name":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6. Physikfeatures berechnen (nur wenn H_tot > 0)
    df["n1"] = df.apply(lambda r: r["n_rpm"] / np.sqrt(r["H_tot"]) if r["H_tot"] > 0 else np.nan, axis=1)
    df["Q1"] = df.apply(lambda r: r["Q_A"] / (r["H_tot"] ** 1.5) if r["H_tot"] > 0 else np.nan, axis=1)

    return df


# === Hauptteil ===
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError("Keine Dateien gefunden – überprüfe den Pfad in DATA_GLOB.")

parts = []
for fp in files:
    try:
        parts.append(parse_one_txt(fp))
    except Exception as e:
        print(f"⚠️ Fehler beim Parsen {fp}: {e}")

master = pd.concat(parts, ignore_index=True)

# Optional: Abrasionsmapping mergen
if os.path.exists(ABRASION_MAP):
    try:
        map_df = pd.read_csv(ABRASION_MAP)
        master = master.merge(map_df, on="stage_id", how="left")
    except Exception as e:
        print(f"⚠️ Abrasions-Mapping konnte nicht geladen werden: {e}")

# Spalten sortieren
base = ["MP_Name", "eta", "n_rpm", "Q_A", "H_tot", "T_Water", "rho", "n1", "Q1", "stage_id"]
extra = [c for c in master.columns if c not in base]
master = master[base + extra]

# CSV schreiben
master.to_csv(OUTPUT_CSV, index=False)
print(f"✔ {OUTPUT_CSV} erstellt ({len(master)} Zeilen, {len(master.columns)} Spalten).")