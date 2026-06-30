#!/usr/bin/env python3
"""
backfill_tennis_results.py — Trägt Sieger für gesammelte Tennis-Paper-Matches nach.

Ergänzt tennis_paper_logger.py: Der Logger sammelt PRE-MATCH-Snapshots gut, sein
eingebautes --settle löst aber NUR über Jupiters EPHEMERE Events-Liste auf —
beendete Matches fallen binnen 1–2 Tagen aus der Liste, bevor der 6h-Settle-Timer
sie greift. Folge: 78 gespielte Matches hingen mit settled=0, der Forward-Paper-Test
war nicht auswertbar.

Dieses Skript füllt actual_winner / pnl_usd / settled aus einer DAUERHAFTEN Quelle:
**Wikipedia** (MediaWiki-API). Tennis-Draw-Seiten markieren den Sieger jedes Matches
per Fettschrift ('''…''') in den Bracket-Templates ({{8TeamBracket}},
{{16TeamBracket-Compact-Tennis3}}, {{4TeamBracket-Tennis3}}, …). Wir parsen
block-isoliert (sonst kollidieren RD1-team1 über mehrere Quali-Brackets), paaren
team(2k-1) vs team(2k) je Runde und nehmen die fette Seite als Sieger.

Namens-Matching wiederverwendet _norm() aus tennis_elo_model (NFKD, diakritikfrei),
mit Nachnamen-Verankerung wie dort.

Quelle ist durabel → dieses Skript ersetzt das kaputte Jupiter-Settle und kann
periodisch laufen (--loop), so wie backfill_football_results.py.

Aufruf:
  python backfill_tennis_results.py --dry           # nur zeigen, nichts schreiben
  python backfill_tennis_results.py                 # ein Durchlauf, schreibt in DB
  python backfill_tennis_results.py --probe-title "2026 Piracicaba Challenger – Singles"
  python backfill_tennis_results.py --loop --interval 21600 --grace-hours 3
"""

import argparse
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests

try:
    import pymssql
except ImportError:
    pymssql = None

from tennis_paper_logger import DB_CONFIG, _paper_pnl

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "boersenbot-research/1.0 (veit.luther@gmx.de) tennis paper-test backfill"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("tennis_backfill")


def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


# ---------------------------------------------------------------- Wikipedia

def wiki_wikitext(title):
    """Roh-Wikitext einer Seite (oder None, wenn sie fehlt)."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "prop": "revisions", "rvslots": "main",
            "rvprop": "content", "titles": title, "format": "json", "redirects": 1,
        }, headers=UA, timeout=25)
        if r.status_code != 200:
            return None
        pages = r.json().get("query", {}).get("pages", {})
        for pg in pages.values():
            if "missing" in pg:
                return None
            revs = pg.get("revisions")
            if revs:
                return revs[0]["slots"]["main"]["*"]
    except Exception as e:
        log.warning(f"  Wikitext-Fetch '{title}' fehlgeschlagen: {e}")
    return None


def wiki_search(query, limit=5):
    """Volltextsuche -> Liste von Seitentiteln (best-effort)."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json",
        }, headers=UA, timeout=25)
        if r.status_code != 200:
            return []
        return [it["title"] for it in r.json().get("query", {}).get("search", [])]
    except Exception as e:
        log.warning(f"  Wiki-Suche '{query}' fehlgeschlagen: {e}")
        return []


# Jupiter nennt Turniere teils nach Sponsor/Ort, Wikipedia nach dem offiziellen
# (anderen) Sponsor- oder Ortsnamen. Substring (lower) -> kanonischer Seiten-Kern.
TOURNAMENT_ALIASES = {
    "targu mures": "INTARO Open",        # Târgu Mureș Challenger -> 2026 INTARO Open
    "eastbourne": "Eastbourne Open",     # "Lexus Eastbourne Open" -> 2026 Eastbourne Open
}


def candidate_titles(tournament, tour, year):
    """Heuristische Seitentitel für ein Turnier + gender-gefilterte Volltextsuche."""
    t = (tournament or "").strip()
    tl = t.lower()
    gender = "Men" if tour == "atp" else "Women"
    cores = [t]
    for key, canon in TOURNAMENT_ALIASES.items():
        if key in tl and canon not in cores:
            cores.insert(0, canon)

    cands = []
    if "wimbledon" in tl and "qualif" in tl:
        cands.append(f"{year} Wimbledon Championships – {gender}'s singles qualifying")
    else:
        for core in cores:
            cands.append(f"{year} {core} – {gender}'s singles")   # kombinierte ATP/WTA-Events
            cands.append(f"{year} {core} – Singles")               # reine Challenger/Single-Gender
            cands.append(f"{year} {core} Challenger – Singles")

    # Suche als Netz (Sponsor-/Diakritika-Fälle); falsches Geschlecht herausfiltern,
    # damit ein kombiniertes Event nicht versehentlich die andere Hälfte zieht.
    for title in wiki_search(f"{year} {t} tennis"):
        low = title.lower()
        if "singles" not in low:
            continue
        is_women = "women" in low
        is_men = "men's" in low and not is_women
        if tour == "atp" and is_women:
            continue
        if tour == "wta" and is_men:
            continue
        cands.append(title)

    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------- Bracket-Parser

def _bracket_blocks(txt):
    """Brace-balancierte {{…Bracket…}}-Blöcke ausschneiden (jeder Block isoliert,
    damit RD1-team1 aus verschiedenen Brackets nicht kollidiert)."""
    blocks, i, n = [], 0, len(txt)
    while True:
        m = re.search(r"\{\{\s*[^|}\n]*Bracket", txt[i:])
        if not m:
            break
        start = i + m.start()
        depth, j = 0, start
        while j < n - 1:
            two = txt[j:j + 2]
            if two == "{{":
                depth += 1
                j += 2
            elif two == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        blocks.append(txt[start:j])
        i = j
    return blocks


def _clean_name(v):
    """team-Param -> reiner Spielername (Sieger-Fett, Flaggen, Wikilinks, Seeds entfernt).
    Bei [[Ziel|Anzeige]] wird das LINKZIEL bevorzugt (Anzeige ist oft abgekürzt)."""
    v = v.strip().strip("'")                                   # Fett-Quotes weg
    v = re.sub(r"\{\{[^{}]*\}\}", "", v)                        # {{flagicon|..}} etc.
    v = re.sub(r"\[\[([^\]]+)\]\]", lambda m: m.group(1).split("|")[0], v)  # Link -> Ziel
    v = re.sub(r"<ref.*?</ref>", "", v, flags=re.S)
    v = re.sub(r"<[^>]+>", "", v)
    v = re.sub(r"''.*?''", "", v)                              # Kursiv-Anmerkungen
    v = re.sub(r"\(.*?\)", "", v)                              # (tennis)-Begriffsklärung
    v = v.replace("&nbsp;", " ").replace("'''", "")
    return re.sub(r"\s+", " ", v).strip()


def _block_pairs(block):
    """(Sieger, Verlierer)-Paare eines Bracket-Blocks aus dem Fett-Marker."""
    by_round = {}    # round -> {slot: (name, is_bold)}
    for line in block.split("\n"):
        m = re.match(r"\s*\|\s*RD(\d+)-team0*(\d+)\s*=\s*(.*)$", line)
        if not m:
            continue
        rnd, slot, raw = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        is_bold = raw.startswith("'''")
        name = _clean_name(raw)
        if name:
            by_round.setdefault(rnd, {})[slot] = (name, is_bold)
    pairs = []
    for slots in by_round.values():
        if not slots:
            continue
        for k in range(1, max(slots) // 2 + 1):
            a, b = slots.get(2 * k - 1), slots.get(2 * k)
            if not a or not b:
                continue
            (an, ab), (bn, bb) = a, b
            if ab and not bb:
                pairs.append((an, bn))
            elif bb and not ab:
                pairs.append((bn, an))
    return pairs


def page_pairs(title):
    txt = wiki_wikitext(title)
    if not txt:
        return []
    pairs = []
    for blk in _bracket_blocks(txt):
        pairs += _block_pairs(blk)
    return pairs


# ---------------------------------------------------------------- Namens-Matching

def _toks(s):
    """Name -> diakritikfreie Kleinbuchstaben-Tokens (Reihenfolge erhalten)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.sub(r"[^a-z0-9]", " ", s).split() if t]


def _tok_match(s, l):
    """Token-Ähnlichkeit mit Initial-Logik: 'S'/'Sandro' oder 'F'/'Felix' gelten als
    Treffer (spätere Bracket-Runden listen Sieger oft als 'F Balshaw' statt voll)."""
    if s == l:
        return 1.0
    if (len(s) == 1 and l.startswith(s)) or (len(l) == 1 and s.startswith(l)):
        return 0.95
    return SequenceMatcher(None, s, l).ratio()


def _containment(ta, tb):
    """Anteil der Tokens des KÜRZEREN Namens, die im längeren ein starkes Pendant
    haben — fängt gekürzte/abgekürzte Markt-Namen ('Lucas Da Silva' ⊂ 'Lucas Andrade
    da Silva', 'S Kopp' ~ 'Sandro Kopp', 'Dal Blanch' ~ 'Dali Blanch')."""
    short, lon = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not short:
        return 0.0
    used, hits = set(), 0
    for s in short:
        best, bi = 0.0, -1
        for i, l in enumerate(lon):
            if i in used:
                continue
            r = _tok_match(s, l)
            if r > best:
                best, bi = r, i
        if best >= 0.85:
            hits += 1
            used.add(bi)
    return hits / len(short)


def name_sim(a, b):
    """Ähnlichkeit zweier Spielernamen — robust gegen (a) Namensumkehr (Token-Set,
    z. B. 'Zhou Yi' vs 'Yi Zhou') und (b) gekürzte Markt-Namen (Containment)."""
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    full = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    tset = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return max(full, tset, _containment(ta, tb))


def resolve_winner(pa, pb, pairs, thresh=0.84):
    """Bestes (Sieger,Verlierer)-Paar für die DB-Paarung finden -> 'A'/'B' oder None."""
    best_side, best_score = None, 0.0
    for w, l in pairs:
        s_aw = min(name_sim(pa, w), name_sim(pb, l))   # A=Sieger
        s_bw = min(name_sim(pa, l), name_sim(pb, w))   # B=Sieger
        if s_aw >= s_bw and s_aw > best_score:
            best_side, best_score = "A", s_aw
        elif s_bw > s_aw and s_bw > best_score:
            best_side, best_score = "B", s_bw
    return (best_side, best_score) if best_score >= thresh else (None, best_score)


# ---------------------------------------------------------------- DB-Lauf

def fetch_pending(conn, grace_hours):
    cur = conn.cursor()
    cur.execute(
        """SELECT event_id, tour, tournament, begin_utc, player_a, player_b,
                  bet_side, entry_price, stake_usd
           FROM bb_TennisPaperBets
           WHERE settled = 0
             AND begin_utc IS NOT NULL
             AND begin_utc < DATEADD(hour, -%d, GETUTCDATE())
           ORDER BY tour, tournament, begin_utc""",
        (int(grace_hours),),
    )
    return cur.fetchall()


def write_winner(conn, event_id, winner, pnl):
    cur = conn.cursor()
    cur.execute(
        """UPDATE bb_TennisPaperBets
           SET actual_winner=%s, pnl_usd=%s, settled=1, updated_utc=GETUTCDATE()
           WHERE event_id=%s""",
        (winner, pnl, event_id),
    )


def run_once(args, conn):
    rows = fetch_pending(conn, args.grace_hours)
    if not rows:
        log.info("Keine offenen Matches zum Nachtragen.")
        return 0
    log.info(f"{len(rows)} offene Match(es).")

    # nach Turnier gruppieren -> jede Wikipedia-Seite nur einmal ziehen
    groups = {}
    for r in rows:
        year = r[3].year if r[3] else datetime.now(timezone.utc).year
        groups.setdefault((r[1], r[2], year), []).append(r)

    page_cache = {}     # title -> pairs
    filled = unresolved = 0
    for (tour, tournament, year), grp in groups.items():
        pairs = []
        used_title = None
        for title in candidate_titles(tournament, tour, year):
            if title not in page_cache:
                page_cache[title] = page_pairs(title)
            if page_cache[title]:
                pairs += page_cache[title]
                used_title = used_title or title
        log.info(f"[{tour.upper()}] {tournament} {year}: {len(grp)} Match(es), "
                 f"{len(pairs)} Bracket-Paare (Seite: {used_title or 'KEINE'})")
        for (event_id, _t, _tn, _b, pa, pb, bet_side, entry, stake) in grp:
            winner, score = resolve_winner(pa, pb, pairs)
            if winner is None:
                unresolved += 1
                log.info(f"    ungelöst: {pa} vs {pb} (bester Score {score:.2f})")
                continue
            pnl = _paper_pnl(bet_side, entry, stake, winner)
            extra = f"  P&L {pnl:+.2f}" if pnl is not None else ""
            log.info(f"    {pa} vs {pb}: Sieger {winner} (Score {score:.2f}){extra}")
            if not args.dry:
                write_winner(conn, event_id, winner, pnl)
            filled += 1
    log.info(f"Durchlauf: {filled} aufgelöst, {unresolved} ungelöst "
             f"{'(dry) ' if args.dry else ''}.")
    return filled


def run(args):
    if args.probe_title:
        pairs = page_pairs(args.probe_title)
        log.info(f"PROBE '{args.probe_title}': {len(pairs)} Paare")
        for w, l in pairs:
            log.info(f"   {w}  schlägt  {l}")
        return

    conn = get_conn()
    while True:
        try:
            run_once(args, conn)
        except Exception as e:
            log.warning(f"Durchlauf-Fehler: {e}")
            try:
                conn = get_conn()
            except Exception:
                pass
        if not args.loop:
            break
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Tennis-Paper-Sieger-Backfill aus Wikipedia (-> Centron).")
    ap.add_argument("--dry", action="store_true", help="Nur zeigen, nicht in DB schreiben")
    ap.add_argument("--loop", action="store_true", help="Dauerlauf (sonst ein Durchlauf)")
    ap.add_argument("--interval", type=int, default=21600, help="Sek. zwischen Durchläufen im --loop (default 6h)")
    ap.add_argument("--grace-hours", type=float, default=3.0,
                    help="Erst Matches nachtragen, deren Anpfiff > grace-hours her ist (default 3)")
    ap.add_argument("--probe-title", help="Debug: eine Wikipedia-Seite parsen, ohne DB")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
