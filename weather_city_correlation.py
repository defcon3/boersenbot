#!/usr/bin/env python3
"""
weather_city_correlation.py — Haengen die Prognosefehler benachbarter Staedte zusammen?

Anlass (Money-Management-Frage 25.07.): Der -1-Autobuy sortiert die Kandidaten
allein nach Preis. Am 25.07. lagen dadurch drei der vier gekauften Positionen in
Ostasien (Taipei, Chengdu, Wuhan). Wenn deren Fehler zusammen kippen, sind das
nicht drei Wetten a 4,81 $, sondern naeherungsweise eine Wette a 14,43 $ - ohne
dass eine Regel das begrenzt.

Diese Auszaehlung beantwortet NUR die Faktenfrage, ob der Klumpen echt ist.
Sie ist keine Strategie und schlaegt keinen Filter vor.

DATENLAGE (der begrenzende Faktor): 13 Zieltage. Die China-Staedte haben 4-7
Tage, ein Paar wie Chengdu-Wuhan also hoechstens 6 gemeinsame. Einzelne
Paar-Korrelationen sind bei n=6 wertlos - der erste Durchlauf lieferte prompt
Tel Aviv/Wellington mit r=+0,59, was geografisch nicht sein kann. Deshalb wird
hier NICHT paarweise geurteilt, sondern ueber alle Paare GEPOOLT und gegen ein
Zufallsmodell getestet.

  A) Fehler-Gleichlauf, gepoolt
     err = settle_k - mu_ens je (Zieltag, Stadt), Lead-1-Snapshot (Vortag,
     identische mu-Definition wie der Autobuy). Mittleres r der Paare INNERHALB
     einer Region gegen das der Paare ZWISCHEN Regionen. Die Interregion-Paare
     dienen zugleich als Rauschband: sie sollten um 0 streuen, und ihre
     Streubreite zeigt, wie gross Zufallsausschlaege bei dieser Stichprobe sind.

  B) Verlust-Kopplung der -1-Klasse, gepoolt + Permutationstest (die Geldfrage)
     Eine -1-Lay verliert, wenn settle_k == k des gelayten Buckets. Gemessen
     wird, wie oft zwei Staedte am selben Zieltag GEMEINSAM verlieren, intra-
     gegen interregional. Signifikanz per Permutation: die Verlust-Flags werden
     INNERHALB jedes Zieltags gemischt. Das erhaelt die Zahl der Verlierer pro
     Tag (also Wetterlage-Tage bleiben Wetterlage-Tage) und zerstoert nur die
     Zuordnung zu Staedten - getestet wird damit ausschliesslich die regionale
     Struktur.

Alles IN-SAMPLE und explorativ: Richtung ja, Groesse nein.
Datenbasis: bb_WeatherLadders (Centron), Zieltage ab 11.07.2026.
"""

import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import pymssql

from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

MIN_COMMON_DAYS = 5     # bewusst niedrig: sonst faellt ganz Ostasien raus
N_PERM = 5000
RNG = np.random.default_rng(20260725)

REGION = {
    "Beijing": "Ostasien", "Busan": "Ostasien", "Chengdu": "Ostasien",
    "Chongqing": "Ostasien", "Guangzhou": "Ostasien", "Hong Kong": "Ostasien",
    "Qingdao": "Ostasien", "Seoul": "Ostasien", "Shanghai": "Ostasien",
    "Shenzhen": "Ostasien", "Taipei": "Ostasien", "Tokyo": "Ostasien",
    "Wuhan": "Ostasien",
    "Amsterdam": "Europa", "Helsinki": "Europa", "London": "Europa",
    "Madrid": "Europa", "Milan": "Europa", "Munich": "Europa", "Moscow": "Europa",
    "Paris": "Europa", "Warsaw": "Europa", "Istanbul": "Europa", "Ankara": "Europa",
    "Karachi": "Suedasien", "Lucknow": "Suedasien", "Kuala Lumpur": "Suedostasien",
    "Manila": "Suedostasien", "Singapore": "Suedostasien",
    "Jeddah": "Naher Osten", "Tel Aviv": "Naher Osten",
    "Buenos Aires": "Suedamerika", "Sao Paulo": "Suedamerika",
    "Mexico City": "Nordamerika", "Panama City": "Nordamerika",
    "Toronto": "Nordamerika", "Cape Town": "Afrika", "Wellington": "Ozeanien",
}


def reg(city):
    return REGION.get(city, "?")


def dedup(df):
    """Am 11.07. liefen ZWEI Ladder-Snapshots -> Zieltag 12.07. sonst doppelt.
    Je (Zieltag, Stadt) bleibt der spaeteste Snapshot stehen."""
    return (df.sort_values("snapshot_utc")
              .drop_duplicates(subset=["target_date", "city"], keep="last"))


def load(cn):
    q = ("SELECT target_date, city, mu_ens, settle_k, snapshot_utc "
         "FROM bb_WeatherLadders "
         "WHERE var=%s AND settle_k IS NOT NULL AND mu_ens IS NOT NULL "
         "  AND CAST(snapshot_utc AS DATE) = DATEADD(day, -1, target_date)")
    return dedup(pd.read_sql(q, cn, params=("max",)).drop_duplicates())


def load_minus1(cn):
    q = ("SELECT target_date, city, k, buy_no, settle_k, snapshot_utc "
         "FROM bb_WeatherLadders "
         "WHERE var=%s AND kind=%s AND offset_fav=-1 AND settle_k IS NOT NULL "
         "  AND CAST(snapshot_utc AS DATE) = DATEADD(day, -1, target_date)")
    return dedup(pd.read_sql(q, cn, params=("max", "eq")))


def messung_a(df):
    print("=" * 78)
    print("A) GLEICHLAUF DER PROGNOSEFEHLER  (err = tatsaechliches Max - Prognose, K)")
    print("=" * 78)
    df = df.copy()
    df["err"] = df["settle_k"] - df["mu_ens"]
    wide = df.pivot_table(index="target_date", columns="city", values="err")
    print(f"{wide.shape[0]} Zieltage x {wide.shape[1]} Staedte "
          f"(nach Dedup der Doppel-Snapshots)\n")

    intra, inter = [], []
    for a, b in combinations(sorted(wide.columns), 2):
        both = wide[[a, b]].dropna()
        if len(both) < MIN_COMMON_DAYS:
            continue
        r = both[a].corr(both[b])
        if pd.isna(r):
            continue
        (intra if reg(a) == reg(b) and reg(a) != "?" else inter).append((r, len(both), a, b))

    if not intra:
        print("Keine bewertbaren Paare innerhalb einer Region -> Frage nicht beantwortbar.")
        return
    mi, me = float(np.mean([r for r, *_ in intra])), float(np.mean([r for r, *_ in inter]))
    sd_inter = float(np.std([r for r, *_ in inter]))
    print(f"   Paare INNERHALB einer Region : n={len(intra):3d}   mittleres r = {mi:+.3f}")
    print(f"   Paare ZWISCHEN den Regionen  : n={len(inter):3d}   mittleres r = {me:+.3f}")
    print(f"   Unterschied                  : {mi - me:+.3f}")
    print(f"\n   Rauschband: die Interregion-Paare - die sachlich unkorreliert sein")
    print(f"   MUESSEN - streuen mit sd={sd_inter:.2f} um {me:+.2f}. Ein Einzelpaar")
    print(f"   muesste also weit ueber r={me + 2 * sd_inter:+.2f} liegen, um mehr als")
    print(f"   Zufall zu sein. Deshalb hier nur die gepoolten Mittel.")

    print(f"\n   Alle Intraregion-Paare (n >= {MIN_COMMON_DAYS} gemeinsame Tage):")
    for r, n, a, b in sorted(intra, reverse=True):
        print(f"      {a:<13} {b:<13} n={n:2d}  r={r:+.2f}   ({reg(a)})")


def messung_b(m1):
    print("\n" + "=" * 78)
    print("B) VERLUST-KOPPLUNG DER -1-KLASSE  (Verlust = Max landet auf dem Bucket)")
    print("=" * 78)
    if m1.empty:
        print("Keine -1-Zeilen gefunden.")
        return
    m1 = m1.copy()
    m1["verlust"] = m1["settle_k"] == m1["k"]
    n, v = len(m1), int(m1["verlust"].sum())
    tage = sorted(m1["target_date"].unique())
    print(f"{n} -1-Kandidaten ueber {len(tage)} Zieltage, davon {v} Verlierer "
          f"({v / n * 100:.1f} %)")
    print("(Das sind ALLE -1-Kandidaten, nicht die gekauften - der Autobuy nimmt "
          "die\nkonservativsten und liegt deshalb deutlich niedriger.)\n")

    per_day = m1.groupby("target_date")["verlust"].agg(["sum", "count"])
    print("Verlierer je Zieltag:")
    for d, row in per_day.iterrows():
        s, c = int(row["sum"]), int(row["count"])
        print(f"   {d}  {s:2d} von {c:2d}  ({s / c * 100:4.1f} %)  {'X' * s}")

    p = v / n
    erwartet_sd = np.mean([np.sqrt(p * (1 - p) / c) for c in per_day["count"]])
    beob_sd = float((per_day["sum"] / per_day["count"]).std())
    print(f"\n   Streuung der Tagesraten: beobachtet {beob_sd * 100:.1f} %-Punkte, "
          f"bei reinem Zufall {erwartet_sd * 100:.1f} %-Punkte erwartet.")
    print("   -> Kein auffaelliges Tages-Clustering." if beob_sd <= erwartet_sd * 1.3
          else "   -> Tagesraten schwanken staerker als der Zufall erklaert.")

    # Gepoolte Koinzidenz: wie oft verlieren zwei Staedte am selben Tag gemeinsam?
    # Matrixform: L (Tage x Staedte) mit 1=Verlust, valid=Stadt an dem Tag im
    # Ladder. L.T @ L zaehlt gemeinsame Verluste je Paar, valid.T @ valid die
    # gemeinsamen Tage - eine Multiplikation statt einer Schleife ueber Paare.
    piv = m1.pivot_table(index="target_date", columns="city", values="verlust")
    staedte = list(piv.columns)
    valid = piv.notna().to_numpy()
    L0 = np.nan_to_num(piv.to_numpy(dtype=float))
    paartage = valid.T.astype(float) @ valid.astype(float)

    gleiche_reg = np.array([[reg(a) == reg(b) and reg(a) != "?" for b in staedte]
                            for a in staedte])
    oben = np.triu(np.ones_like(paartage, dtype=bool), k=1)
    genug = paartage >= MIN_COMMON_DAYS
    m_intra = oben & genug & gleiche_reg
    m_inter = oben & genug & ~gleiche_reg

    def quote(L, maske):
        co = L.T @ L
        n = paartage[maske].sum()
        return (co[maske].sum(), n)

    gi, ni = quote(L0, m_intra)
    ge, ne = quote(L0, m_inter)
    if not ni:
        print("\n   Keine Intraregion-Paare mit genug gemeinsamen Tagen.")
        return
    print(f"\n   Gemeinsame Verlusttage, gepoolt ueber alle Paare:")
    print(f"      innerhalb einer Region: {int(gi):3d} von {int(ni):4d} Paartagen "
          f"= {gi / ni * 100:.1f} %   ({int(m_intra.sum())} Paare)")
    print(f"      zwischen den Regionen : {int(ge):3d} von {int(ne):4d} Paartagen "
          f"= {ge / ne * 100:.1f} %   ({int(m_inter.sum())} Paare)")
    print(f"      bei Unabhaengigkeit erwartet: {p * p * 100:.1f} %")

    # Permutation: Verlust-Flags INNERHALB jedes Zieltags mischen.
    beob = gi / ni
    null = np.empty(N_PERM)
    tag_idx = [np.where(valid[t])[0] for t in range(L0.shape[0])]
    L = L0.copy()
    for i in range(N_PERM):
        for t, idx in enumerate(tag_idx):
            L[t, idx] = RNG.permutation(L0[t, idx])
        a_, b_ = quote(L, m_intra)
        null[i] = a_ / b_ if b_ else np.nan
    pval = float((null >= beob).mean())
    print(f"\n   Permutationstest ({N_PERM} Mischungen, Verlierer pro Tag bleiben "
          f"erhalten):")
    print(f"      beobachtet {beob * 100:.1f} %, Zufallsmittel {np.nanmean(null) * 100:.1f} %, "
          f"p = {pval:.3f}")
    if pval < 0.05:
        print("      -> Regionale Kopplung nachweisbar.")
    elif pval < 0.20:
        print("      -> Tendenz sichtbar, aber nicht belastbar (zu wenige Tage).")
    else:
        print("      -> Kein Nachweis einer regionalen Kopplung in diesen Daten.")

    print("\n   Tage mit mehreren Verlierern:")
    for d, row in per_day.iterrows():
        if row["sum"] >= 2:
            regs = defaultdict(list)
            for c in m1[(m1["target_date"] == d) & m1["verlust"]]["city"]:
                regs[reg(c)].append(c)
            teile = [f"{r}: {', '.join(sorted(cs))}" + (" <<" if len(cs) >= 2 else "")
                     for r, cs in sorted(regs.items())]
            print(f"      {d}  " + " | ".join(teile))


def main():
    cn = pymssql.connect(**DB_CONFIG)
    try:
        df, m1 = load(cn), load_minus1(cn)
    finally:
        cn.close()
    if df.empty:
        print("Keine Lead-1-Zeilen mit Settlement gefunden.")
        return 1
    messung_a(df)
    messung_b(m1)
    print("\n" + "-" * 78)
    print("13 Zieltage, IN-SAMPLE, explorativ. '<<' markiert zwei Verlierer derselben "
          "Region\nam selben Tag - genau das Muster, um das es beim Klumpen geht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
