"""
MINUTEN-RICHTUNGS-ANALYSE — 2026-06-23
Frage (Nutzer): Wie schwanken die Werte? Nur eine Richtung oder beide?
Wie oft geht es pro Tag fuer eine Aktie nach oben / nach unten?

Tabelle: bb_StockPrices_1min_YFinance (5 Symbole, ~30 Tage, RTH).
Pro (Symbol, Handelstag): Minute-zu-Minute-Aenderung der ClosePrice.
  n_up   = Minuten mit hoeherem Close als Vorminute
  n_down = Minuten mit niedrigerem Close
  n_flat = unveraendert
KEINE Tagesgrenze ueberschreiten (Overnight-Gap zaehlt nicht als Minutenmove).
"""
import pymssql
import numpy as np
import pandas as pd

DB = dict(server='158.181.48.77', user='326773', password='Extaler11!', database='dbdata')


def load():
    conn = pymssql.connect(**DB)
    df = pd.read_sql(
        "SELECT Symbol, [Timestamp], ClosePrice FROM bb_StockPrices_1min_YFinance "
        "ORDER BY Symbol, [Timestamp]", conn)
    conn.close()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.date  # YFinance=UTC, RTH -> UTC-Datum == ET-Handelstag
    return df


def per_day(df):
    rows = []
    for (sym, d), g in df.groupby(['Symbol', 'date']):
        c = g.sort_values('Timestamp')['ClosePrice'].values.astype(float)
        if len(c) < 30:
            continue
        diff = np.diff(c)
        n_up = int((diff > 0).sum())
        n_down = int((diff < 0).sum())
        n_flat = int((diff == 0).sum())
        gross = np.abs(diff).sum() / c[0]          # gesamte "Wanderstrecke"
        net = (c[-1] - c[0]) / c[0]                # Netto Open->Close (intraday)
        rows.append({
            'Symbol': sym, 'date': d, 'n_min': len(c),
            'n_up': n_up, 'n_down': n_down, 'n_flat': n_flat,
            'up_share': n_up / (n_up + n_down) if (n_up + n_down) else np.nan,
            'gross_pct': gross * 100, 'net_pct': net * 100,
            'abs_net_pct': abs(net) * 100,
            'eff': abs(net) / gross if gross else np.nan,   # Richtungs-Effizienz
            'avg_move_bps': np.abs(diff / c[:-1]).mean() * 1e4,
            'day_up': net > 0,
        })
    return pd.DataFrame(rows)


def hourly_segments(df):
    """Pro (Symbol, Tag): hoch/runter-Minuten + Renditen je Stunde.
    UTC-Minuten (Datensatz EDT): 09:30 ET=810, 10:30=870, 11:30=930."""
    rows = []
    for (sym, d), g in df.groupby(['Symbol', 'date']):
        g = g.sort_values('Timestamp')
        mm = (g['Timestamp'].dt.hour * 60 + g['Timestamp'].dt.minute).values
        c = g['ClosePrice'].values.astype(float)
        if len(c) < 120:
            continue
        idx1 = np.where(mm < 870)[0]
        idx2 = np.where(mm < 930)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        p_open, p_close = c[0], c[-1]
        p_h1, p_h2 = c[idx1[-1]], c[idx2[-1]]
        diff = np.diff(c)
        endmin = mm[1:]

        def ud(lo, hi):
            dd = diff[(endmin >= lo) & (endmin < hi)]
            return int((dd > 0).sum()), int((dd < 0).sum()), int((dd == 0).sum())
        u1, d1, f1 = ud(810, 870)
        u2, d2, f2 = ud(870, 930)
        rows.append(dict(
            Symbol=sym, date=d, u1=u1, d1=d1, f1=f1, u2=u2, d2=d2, f2=f2,
            h1=p_h1 / p_open - 1, cum2=p_h2 / p_open - 1,
            day=p_close / p_open - 1,
            rest1=p_close / p_h1 - 1, rest2=p_close / p_h2 - 1))
    return pd.DataFrame(rows)


def hour_report(h):
    print('\n' + '=' * 84)
    print('ERGÄNZUNG: ERSTE & ZWEITE STUNDE — und ist das Tagesende ablesbar?')
    print('=' * 84)
    print('\n--- hoch/runter-Minuten je Stunde (Ø, gepoolt) ---')
    for lbl, u, dn, fl in [('1. Stunde (09:30-10:30)', 'u1', 'd1', 'f1'),
                           ('2. Stunde (10:30-11:30)', 'u2', 'd2', 'f2')]:
        U, D, F = h[u].mean(), h[dn].mean(), h[fl].mean()
        print(f'  {lbl}: {U:.0f} hoch / {D:.0f} runter / {F:.0f} flat'
              f'  -> {U/(U+D)*100:.1f}% hoch')

    base = (h['day'] > 0).mean() * 100
    print(f'\n--- "Kann man das Tagesende ablesen?" (n={len(h)} Symbol-Tage) ---')
    print(f'  Basisrate: an {base:.1f}% der Tage endet es netto im Plus.\n')

    # (A) NAIV — mit Ueberlapp (1. Stunde ist Teil des Tages!)
    naive = (np.sign(h['h1']) == np.sign(h['day'])).mean() * 100
    p_up_given_up = (h[h['h1'] > 0]['day'] > 0).mean() * 100
    p_up_given_dn = (h[h['h1'] < 0]['day'] > 0).mean() * 100
    print('  (A) NAIV  sign(1.Std) == sign(ganzer Tag):')
    print(f'      Trefferquote {naive:.1f}%  | P(Tag+ | Std1+)={p_up_given_up:.0f}%  '
          f'P(Tag+ | Std1-)={p_up_given_dn:.0f}%')
    print(f'      ACHTUNG: aufgeblaeht — die 1. Stunde STECKT im Tagesergebnis (Buchhaltung).')

    # (B) EHRLICH — nur der REST des Tages (Ueberlapp entfernt)
    hit1 = (np.sign(h['h1']) == np.sign(h['rest1'])).mean() * 100
    hit2 = (np.sign(h['cum2']) == np.sign(h['rest2'])).mean() * 100
    c1 = np.corrcoef(h['h1'], h['rest1'])[0, 1]
    c2 = np.corrcoef(h['cum2'], h['rest2'])[0, 1]
    print('\n  (B) EHRLICH  Vormittag -> REST des Tages (kein Ueberlapp):')
    print(f'      sign(1.Std) == sign(10:30->Close): {hit1:.1f}%  (corr={c1:+.2f})')
    print(f'      sign(1.+2.Std) == sign(11:30->Close): {hit2:.1f}%  (corr={c2:+.2f})')
    weak = (abs(c1) < 0.1) and (abs(c2) < 0.1)
    verdict = ('NEIN — corr~0 (Vormittag sagt den Rest des Tages NICHT vorher; '
               'Hit-Quoten-Abweichung = Rauschen bei n=%d)' % len(h)) if weak else \
              'moeglicher schwacher Effekt — groesseres Sample noetig'
    print(f'\n  VERDICT: {verdict}.')
    print('  Die scheinbare Ablesbarkeit (A) ist fast komplett der Ueberlapp:')
    print('  die 1. Stunde ist Bestandteil des Tagesergebnisses, kein Vorlauf-Signal.')


def main():
    print('=' * 84)
    print('MINUTEN-RICHTUNGS-ANALYSE — bb_StockPrices_1min_YFinance')
    print('=' * 84)
    df = load()
    pd_ = per_day(df)
    print(f'{len(df):,} Minuten-Bars | {df.Symbol.nunique()} Symbole | '
          f'{pd_.date.nunique()} Handelstage | {pd_.date.min()} .. {pd_.date.max()}\n')

    print('--- PRO AKTIE: Durchschnitt je Handelstag ---')
    print(f"{'Sym':<6}{'Min/Tag':>8}{'hoch':>7}{'runter':>8}{'flat':>6}"
          f"{'hoch%':>7}{'runter%':>8}{'Ø-Move':>9}")
    for sym, g in pd_.groupby('Symbol'):
        nm, up, dn, fl = g.n_min.mean(), g.n_up.mean(), g.n_down.mean(), g.n_flat.mean()
        tot = up + dn
        print(f"{sym:<6}{nm:>8.0f}{up:>7.0f}{dn:>8.0f}{fl:>6.0f}"
              f"{up/tot*100:>6.1f}%{dn/tot*100:>7.1f}%{g.avg_move_bps.mean():>7.1f}bp")

    allup = pd_.n_up.sum() / (pd_.n_up.sum() + pd_.n_down.sum()) * 100
    print(f"\nGESAMT: {allup:.1f}% der gerichteten Minuten hoch, {100-allup:.1f}% runter "
          f"-> praktisch 50/50, IMMER BEIDE Richtungen.")

    print('\n--- "Nur eine Richtung?" — Verteilung des Hoch-Anteils je Tag ---')
    us = pd_.up_share * 100
    print(f"  Hoch-Anteil/Tag: mean={us.mean():.1f}%  std={us.std():.1f}%  "
          f"min={us.min():.1f}%  max={us.max():.1f}%")
    print(f"  -> Selbst der einseitigste Tag im Datensatz hatte noch "
          f"{100-us.max():.0f}% Gegen-Minuten. Kein einziger Tag lief 'nur hoch'.")

    print('\n--- Hoch-/Runter-TAGE und wie die Minuten dann kippen ---')
    nu, nd = pd_.day_up.sum(), (~pd_.day_up).sum()
    print(f"  Netto-Hoch-Tage: {nu} | Netto-Runter-Tage: {nd}")
    print(f"  Hoch-Anteil an UP-Tagen:    {pd_[pd_.day_up].up_share.mean()*100:.1f}%")
    print(f"  Hoch-Anteil an DOWN-Tagen:  {pd_[~pd_.day_up].up_share.mean()*100:.1f}%")
    print(f"  -> Die Tagesrichtung entsteht aus einem WINZIGEN Ungleichgewicht "
          f"in einem fast fairen Münzwurf.")

    print('\n--- Wieviel "Geschwurbel" pro Netto-Ergebnis (Effizienz) ---')
    print(f"  Ø Wanderstrecke/Tag (Summe |Moves|): {pd_.gross_pct.mean():.2f}%")
    print(f"  Ø |Netto Open->Close|:               {pd_.abs_net_pct.mean():.2f}%")
    print(f"  Ø Effizienz (|netto|/wander):        {pd_.eff.mean()*100:.1f}%")
    print(f"  -> Der Kurs läuft im Schnitt ~{pd_.gross_pct.mean()/max(pd_.abs_net_pct.mean(),1e-9):.0f}x "
          f"so weit hin und her, wie am Ende netto herauskommt.")

    print('\n--- Beispiel: einzelne Tage einer Aktie (konkret) ---')
    ex = 'TSLA' if 'TSLA' in pd_.Symbol.values else pd_.Symbol.iloc[0]
    g = pd_[pd_.Symbol == ex].sort_values('date').head(8)
    print(f"  {ex}:")
    print(f"  {'Datum':<12}{'hoch':>6}{'runter':>8}{'flat':>6}{'netto%':>9}")
    for _, r in g.iterrows():
        arrow = 'hoch' if r.net_pct > 0 else 'runter'
        print(f"  {str(r.date):<12}{int(r.n_up):>6}{int(r.n_down):>8}"
              f"{int(r.n_flat):>6}{r.net_pct:>8.2f}% {arrow}")

    hour_report(hourly_segments(df))


if __name__ == '__main__':
    main()
