#!/usr/bin/env python3
"""
GDELT MARKT-STIMMUNGS-ETL (Option C, 2026-05-18).

Baut eine markt-weite News-Tagesstimmung aus dem GDELT-2.0-GKG-Bulkfeed
auf (data.gdeltproject.org ist vom VPS erreichbar; die Live-API ist
ratenlimitiert -> Bulk ist ohnehin die richtige, backtestbare Quelle).

ZWECK: kleinster sinnvoller Test der Frage „bringt News-Sentiment als
Conditioner auf den Down-Streak ueberhaupt etwas?" — analog zum VIX-Test
(cc_meanrev_vix.py). NICHT pro Aktie (zu duenn), sondern MARKT-WEIT.

DESIGN / EHRLICHE APPROXIMATIONEN (bewusst dokumentiert):
- GDELT 2.0 GKG ab 2015-02-18 (15-Min-Slots). Voller Feed = ~2 TB/Jahr,
  unnoetig: Markt-Tone ist ein Mittel ueber tausende Artikel und
  langsam. Wir SAMPELN SLOTS_PER_DAY Slots/Tag (Default 6, alle 4 h)
  und mitteln. Sampling-Rauschen wird mitgeschrieben (n_art, std), damit
  der Backtest die Messunsicherheit kennt — keine versteckte Annahme.
- Pro Tag: Mittel-Tone ALLER Artikel UND nur oekonomischer (V1THEMES
  enthaelt 'ECON_'); beide Spalten -> zwei Conditioner-Varianten.
- Tone = GKG-Spalte 15 (V1.5TONE), erstes Komma-Feld (avg tone).
- RESUMABLE: schreibt eine Zeile je GDELT-Datum, ueberspringt bereits
  vorhandene Daten; chronologisch alt->neu (alte IS-Periode zuerst
  nutzbar). Robust gegen fehlende/halbe Slots (Retry, dann skip+log).
- LOOK-AHEAD: hier nur Rohdaten je Publikationsdatum. Die Lag-Disziplin
  (Entscheidung Close t nutzt Tone t-1 / Trailing) macht der BACKTEST.

Output: gdelt_market_tone.csv  (date,n_slots,n_art_all,tone_all_mean,
tone_all_std,n_art_econ,tone_econ_mean). Lauf:
  nohup venv/bin/python -u gdelt_market_tone_etl.py \
        > gdelt_market_tone_etl.log 2>&1 &
"""
import sys, io, os, csv, time, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests

START   = dt.date(2015, 2, 19)            # GDELT 2.0 sicher verfuegbar
END     = dt.date.today() - dt.timedelta(days=1)
SLOTS_PER_DAY = 6                          # gesampelte 15-Min-Slots/Tag
SLOTS   = ["%02d0000" % h for h in range(0, 24, 24 // SLOTS_PER_DAY)]
OUT     = "/home/veit/boersenbot/gdelt_market_tone.csv"
BASEURL = "http://data.gdeltproject.org/gdeltv2/%s%s.gkg.csv.zip"
HDR     = {"User-Agent": "Mozilla/5.0 (research; market-tone-etl)"}
COLS    = ["date", "n_slots", "n_art_all", "tone_all_mean",
           "tone_all_std", "n_art_econ", "tone_econ_mean"]


def fetch_slot(datestr, slot, tries=3):
    """Ein GKG-Slot -> Liste tone-Werte (all) und (econ). [] bei Ausfall."""
    url = BASEURL % (datestr, slot)
    for k in range(tries):
        try:
            r = requests.get(url, headers=HDR, timeout=60)
            if r.status_code == 404:
                return None, None                  # Slot existiert nicht
            r.raise_for_status()
            if len(r.content) < 200:
                raise ValueError("leer/zu klein")
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            nm = zf.namelist()[0]
            ta, te = [], []
            rd = csv.reader(io.TextIOWrapper(zf.open(nm), encoding="latin-1",
                                             errors="ignore"), delimiter="\t")
            for row in rd:
                if len(row) < 16:
                    continue
                try:
                    tone = float(row[15].split(",")[0])
                except (ValueError, IndexError):
                    continue
                ta.append(tone)
                if "ECON_" in row[7]:
                    te.append(tone)
            return ta, te
        except Exception as ex:
            if k == tries - 1:
                print(f"   ! {datestr} {slot} fail: {ex}", flush=True)
                return [], []
            time.sleep(1.5 * (k + 1))
    return [], []


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done.add(row["date"])
        print(f"Resume: {len(done)} Tage bereits vorhanden.", flush=True)
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(COLS); fh.flush()

    d = START
    n_done = 0
    while d <= END:
        ds = d.strftime("%Y%m%d")
        if ds in done:
            d += dt.timedelta(days=1); continue
        all_t, econ_t, ok_slots = [], [], 0
        with ThreadPoolExecutor(max_workers=len(SLOTS)) as ex:
            res = list(ex.map(lambda s: fetch_slot(ds, s), SLOTS))
        for ta, te in res:
            if ta is None:                          # 404 -> Slot fehlt
                continue
            if ta:
                ok_slots += 1
                all_t.extend(ta); econ_t.extend(te)
        if ok_slots == 0:
            print(f"  {ds}: kein Slot -> spaeter erneut", flush=True)
            d += dt.timedelta(days=1); continue
        n = len(all_t)
        mean_a = sum(all_t) / n
        var_a = sum((x - mean_a) ** 2 for x in all_t) / n if n > 1 else 0.0
        ne = len(econ_t)
        mean_e = (sum(econ_t) / ne) if ne else ""
        w.writerow([ds, ok_slots, n, f"{mean_a:.4f}", f"{var_a**0.5:.4f}",
                    ne, (f"{mean_e:.4f}" if ne else "")])
        fh.flush()
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... {ds} | {n_done} neue Tage | Slots {ok_slots} "
                  f"| n_art {n} | tone {mean_a:+.3f}", flush=True)
        d += dt.timedelta(days=1)

    fh.close()
    print(f"FERTIG: {n_done} neue Tage geschrieben -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
