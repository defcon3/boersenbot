#!/usr/bin/env python3
"""
Hot-Wallet-Generator für den Jupiter-Prediction-Bot.

Erzeugt EINEN dedizierten Solana-Keypair, der NUR das Trade-Kapital hält
(niemals die Phantom-Hauptwallet). Der Private Key wird ausschließlich in
lokale, gitignored Dateien geschrieben — NIE in die Konsole, damit er nicht
in Logs, Shell-History oder Chat-Transcripts landet.

Ausgabedateien (alle unter hot_wallet/, via .gitignore ausgeschlossen):
  keypair.json        -> 64-Byte-Array (Solana-CLI-Format; der Bot lädt das zum Signieren)
  phantom_import.txt  -> Base58 des Secret Keys (zum Import in Phantom)
  address.txt         -> öffentliche Adresse (unkritisch)

SICHERHEIT:
  - Läuft NICHT, wenn bereits ein Keypair existiert (sonst Verlust einer
    eventuell schon gefundeten Wallet). Zum Neuerzeugen erst hot_wallet/ löschen.
  - Auf dem VPS später: chmod 600 hot_wallet/keypair.json
"""

import json
import os
import stat
import sys
from pathlib import Path

import base58
from solders.keypair import Keypair

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

WALLET_DIR = Path("hot_wallet")
KEYPAIR_FILE = WALLET_DIR / "keypair.json"
PHANTOM_FILE = WALLET_DIR / "phantom_import.txt"
ADDRESS_FILE = WALLET_DIR / "address.txt"


def _lock_down(path: Path):
    """Datei auf nur-Eigentümer-Lesen/Schreiben setzen (greift v.a. auf Linux/VPS)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except Exception:
        pass  # Windows ignoriert das weitgehend — auf dem VPS setzen wir es erneut.


def main():
    # --- Sicherheits-Stopp: vorhandenen Key niemals überschreiben ---
    if KEYPAIR_FILE.exists():
        print("ABBRUCH: hot_wallet/keypair.json existiert bereits.")
        print("Es wird NICHTS überschrieben (sonst Verlust des Zugriffs auf evtl.")
        print("bereits eingezahltes Geld). Zum Neuerzeugen erst den Ordner")
        print("hot_wallet/ manuell löschen — nur wenn die Wallet garantiert leer ist.")
        sys.exit(1)

    WALLET_DIR.mkdir(exist_ok=True)

    # --- Keypair erzeugen ---
    kp = Keypair()
    secret_bytes = bytes(kp)            # 64 Byte: seed(32) + pubkey(32)
    pubkey_str = str(kp.pubkey())       # Base58-Adresse
    phantom_b58 = base58.b58encode(secret_bytes).decode()  # Phantom-Import-Format

    # --- Schreiben (Private Key NUR in Dateien) ---
    KEYPAIR_FILE.write_text(json.dumps(list(secret_bytes)), encoding="utf-8")
    PHANTOM_FILE.write_text(phantom_b58 + "\n", encoding="utf-8")
    ADDRESS_FILE.write_text(pubkey_str + "\n", encoding="utf-8")
    for f in (KEYPAIR_FILE, PHANTOM_FILE):
        _lock_down(f)

    # --- Verifikation: Datei wieder laden, Pubkey muss übereinstimmen ---
    reloaded = Keypair.from_bytes(bytes(json.loads(KEYPAIR_FILE.read_text())))
    assert str(reloaded.pubkey()) == pubkey_str, "Verifikation fehlgeschlagen!"

    # --- Konsole: NUR öffentliche Infos ---
    print("=" * 60)
    print("Hot-Wallet erzeugt ✓  (Private Key NUR in lokalen Dateien)")
    print("=" * 60)
    print()
    print(f"Öffentliche Adresse (hier Geld hinschicken):")
    print(f"  {pubkey_str}")
    print()
    print("Geheime Dateien (gitignored, NIEMALS teilen/committen):")
    print(f"  {KEYPAIR_FILE}   -> für den Bot (Signieren)")
    print(f"  {PHANTOM_FILE}   -> Base58-Key zum Import in Phantom")
    print()
    print("Verifikation: Keypair erfolgreich zurückgeladen, Pubkey stimmt.")
    print()
    print("NÄCHSTE SCHRITTE:")
    print("  1. phantom_import.txt LOKAL öffnen, Inhalt in Phantom importieren")
    print("     (Phantom: Konto hinzufügen -> Privaten Schlüssel importieren)")
    print("  2. Nur das Trade-Kapital (USDC) + ~0,02 SOL für Gas an die Adresse")
    print("     oben senden — NICHT mehr, als du für diesen Trade riskieren willst.")
    print("  3. Datei phantom_import.txt danach sicher löschen (Key bleibt in keypair.json).")


if __name__ == "__main__":
    main()
