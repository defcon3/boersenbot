# Domain-Umzug weg von Centron / internet1 — Action-Plan

**Stand:** 05.08.2026
**Anlass:** Centron (13 €/Monat) soll perspektivisch fallen
(s. `BUNDESLIGA_MIGRATION_PLAN.md`). Vorher muss geklärt sein, was mit den
Domains passiert.

**Kernbefund:** `veitluther.de` hängt an Centrons Nameservern, obwohl die Seite
auf dem Contabo-VPS läuft. Eine Kündigung bei Centron nimmt die DNS-Zone mit —
die Seite wäre offline, obwohl der VPS unverändert weiterläuft. Und falls die
Domains dort auch registriert sind (sehr wahrscheinlich, s. Phase 0), werden sie
zum Vertragsende **gelöscht** und sind danach für jeden frei registrierbar.

---

## Ist-Zustand (gemessen 05.08.2026, nicht angenommen)

Autoritative Nameserver **beider** Domains: `ns.internet1.de`, `ns2`, `ns3`
(SOA-Admin `hostmaster.internet1.de`).

### Zone `veitluther.de` — SOA-Serial 2026051504

| Record | Typ | Ziel | Muss mit? |
|---|---|---|---|
| `veitluther.de` | A | 144.91.98.234 (Contabo) | **ja** |
| `www.veitluther.de` | A | 144.91.98.234 (Contabo) | **ja** |
| `stats.veitluther.de` | A | 188.64.60.179 (Centron-Statistik) | nein — stirbt mit dem Vertrag |

### Zone `frau-von-allerliebst.de` — SOA-Serial 2016040401 (seit 2016 unverändert)

| Record | Typ | Ziel | Muss mit? |
|---|---|---|---|
| `frau-von-allerliebst.de` | A | 158.181.48.160 (Centron-IIS) | ja, bis ASPX umgezogen |
| `www.frau-von-allerliebst.de` | A | 158.181.48.160 (Centron-IIS) | ja, bis ASPX umgezogen |
| `stats.frau-von-allerliebst.de` | A | 188.64.60.179 | nein |

**Kein MX, kein TXT/SPF/DMARC, kein CAA, kein SRV** in beiden Zonen. Es gehen
also **keine E-Mail-Postfächer** verloren, und der Zone-Nachbau ist trivial:
zwei A-Records je Domain.

### TLS

Let's Encrypt auf dem VPS, `authenticator = nginx` ⇒ **HTTP-01**, nicht DNS-01.
Die Erneuerung hängt damit **nicht** am DNS-Provider, sondern nur daran, dass
der A-Record auf 144.91.98.234 zeigt. Zertifikat `veitluther.de` +
`www.veitluther.de` gültig bis **12.10.2026**, `certbot.timer` aktiv.
nginx kennt auf dem VPS ausschließlich `veitluther.de www.veitluther.de`.

### Was DENIC-RDAP hergibt

`frau-von-allerliebst.de`: Status `active`, „last changed" **18.07.2014**.
Der **Registrar wird seit der DSGVO nicht mehr veröffentlicht** — RDAP liefert
eine leere `entities`-Liste. Wer die Domains hält, steht nur im Centron-Panel
bzw. auf der Rechnung.

---

## Phase 0 — Klären, bevor irgendetwas angefasst wird

**Erledigt 05.08.2026:** Beide Domains sind laut Kundenpanel bei Centron
hinterlegt. Eine **Kündigungsfrist ist im Panel nicht erkennbar** — das heißt
nicht, dass es keine gibt.

### Was die AGB sagen (centron.de/agb, Abschnitt 6d)

Die Frist staffelt sich nach der Vertragslaufzeit:

| Laufzeit | Kündigungsfrist |
|---|---|
| 1 Monat | 1 Woche zum Laufzeitende |
| 3 Monate | 2 Wochen |
| 6 Monate | 1 Monat |
| **12 Monate** | **3 Monate** |
| 24+ Monate | 6 Monate |

Bei der üblichen 12-Monats-Laufzeit sind das **drei Monate zum Laufzeitende** —
verpasst man den Termin, läuft der Vertrag ein weiteres Jahr. Das ist die
kritische Zahl des ganzen Plans.

**Möglicher Einwand zugunsten des Nutzers:** Seit dem *Gesetz für faire
Verbraucherverträge* (01.03.2022) dürfen sich Verbraucherverträge über
wiederkehrende Leistungen nach der Erstlaufzeit nur noch auf unbestimmte Zeit
verlängern und sind dann mit **höchstens einem Monat** kündbar; für online
geschlossene Verbraucherverträge gilt zudem die Kündigungsbutton-Pflicht
(seit 01.07.2022). Die AGB-Staffel oben wirkt wie alter Stand. Ob das greift,
hängt daran, ob der Vertrag als **Verbraucher-** oder **Geschäftskundenvertrag**
läuft — bei letzterem gilt die Staffel. Keine Rechtsberatung; im Zweifel
schriftlich geltend machen.

**Ebenfalls aus den AGB (Abschnitt 4):** Centron ist bei Domains ausdrücklich
nur **Vermittler**, nicht Registry. Der Wegzug ist damit ein normaler
Providerwechsel; zu AuthInfo/KK und zur Domain-Löschung bei Vertragsende sagen
die AGB **nichts** — deshalb schriftlich anfragen (s. Phase 1).

- [ ] **Vertragsende + Frist schriftlich** bei `info@centron.de` erfragen.
      In derselben Mail gleich die AuthInfo-Codes anfordern (Phase 1) — das ist
      **keine** Kündigung und löst keine Frist aus.
- [ ] Prüfen, ob die Domains **Inklusivleistung** des Hosting-Pakets sind. Falls
      ja: ein Wegzug der Domains senkt den Preis nicht — er entkoppelt nur das
      Risiko. Das ist trotzdem richtig so.

---

## Phase 1 — Domains sichern (unabhängig von allem anderen, zuerst)

Diese Phase ist **komplett entkoppelt** von der ASPX- und DB-Migration. Ein
Domain-Transfer ändert nur, wer die Domain verwaltet — die Seiten laufen
unverändert weiter, solange die A-Records gleich bleiben. Deshalb kann und soll
das sofort passieren, lange bevor gekündigt wird.

**Registrar-Entscheidung 05.08.2026: INWX.** Domain-Spezialist, ~6 €/Jahr je
`.de`, DNS-Panel mit API (skriptbare Records). Preisvergleich, der dahinter
steht (Vergleichsportale, Stand 08/2026): Netcup ~5 €, INWX ~6 €, Contabo ~14 €
je Domain und Jahr; Infomaniak im `.de`-Vergleich nicht gelistet.

### 1a — `kreativkommo.de` als Testlauf (kein Risiko)

`kreativkommo.de` ist **frei** — geprüft 05.08.2026: DENIC-RDAP antwortet 404,
alle DNS-Typen NXDOMAIN. Kein Halter, keine Zone, kein Kauf nötig.

Weil sie keine Historie und keinen laufenden Betrieb hat, ist sie der
gefahrlose Probelauf für das neue Panel — **vor** dem Transfer der echten
Domains. Geht dabei etwas schief, ist nichts verloren.

- [ ] `kreativkommo.de` bei INWX **registrieren** (Neuregistrierung, kein Transfer).
- [ ] **Nicht bei Centron registrieren** — sonst liegen drei statt zwei Domains
      in dem Vertrag, der weg soll.
- [ ] Im INWX-DNS testweise einen A-Record auf 144.91.98.234 setzen und die
      Auflösung prüfen. Damit ist das Panel erprobt, bevor es ernst wird.
- [ ] Offen: wofür `kreativkommo.de` verwendet wird und ob sie dauerhaft auf den
      VPS zeigt (dann nginx-`server_name` + eigenes Zertifikat nötig).

### 1b — die beiden echten Domains nachziehen

- [ ] **TTLs bei Centron senken** auf 300 s, mindestens 24 h vor dem Umzug.
      Falls die Oberfläche das nicht hergibt: überspringen, die Zonen sind so
      klein, dass ein längerer Übergang verkraftbar ist.
- [ ] **Zonen bei INWX vorab anlegen** — die vier A-Records aus der Tabelle
      oben, `stats.*` bewusst weglassen. Noch nicht aktivieren.
- [ ] **AuthInfo-Code** für beide Domains bei Centron anfordern (zusammen mit
      der Fristanfrage aus Phase 0). Muss herausgegeben werden; bei `.de`
      üblicherweise sofort im Panel.
- [ ] **KK-Antrag** bei INWX stellen — als *Providerwechsel*, nicht als
      Neuregistrierung. Zuerst `frau-von-allerliebst.de`, dann bei Erfolg
      `veitluther.de` — so hängt nie der Live-Betrieb an einem ungeprüften Schritt.
- [ ] **Nach dem Transfer verifizieren:**
      `Resolve-DnsName veitluther.de -Type NS -Server 8.8.8.8` zeigt die neuen NS,
      und die A-Records beider Domains stimmen noch.
      Danach: `https://veitluther.de` und `https://frau-von-allerliebst.de/bundesliga/`
      im Browser prüfen.
- [ ] **Certbot-Renewal einmal trocken testen:**
      `sudo certbot renew --dry-run` auf dem VPS. Muss durchlaufen, sonst
      stirbt das Zertifikat still am 12.10.

**Ergebnis dieser Phase:** Die Kündigung bei Centron kann danach niemanden mehr
die Domains kosten. Ab hier ist der Rest reine Terminplanung.

---

## Phase 2 — Inhalte von Centron holen

Läuft nach `BUNDESLIGA_MIGRATION_PLAN.md` (Commit `f11cecab`): 18 ASPX-Seiten →
Flask, MSSQL → PostgreSQL. Die Seiten sind alle **rein lesend** ⇒ Parallelbetrieb
ist gefahrlos.

- [ ] Flask-Nachbau auf dem Contabo-VPS unter einer Testadresse hochziehen.
- [ ] **Parallelbetrieb**: alte ASPX-Seite bleibt online, bis der Nachbau steht.
- [ ] Umschalten = `frau-von-allerliebst.de` A-Record von 158.181.48.160 auf
      144.91.98.234 ziehen. Das geht nach Phase 1 beim neuen Provider in einer
      Minute und ist jederzeit zurückdrehbar.
- [ ] nginx auf dem VPS um `server_name frau-von-allerliebst.de
      www.frau-von-allerliebst.de` erweitern, **Zertifikat mit ausstellen**
      (`certbot --nginx -d frau-von-allerliebst.de -d www.frau-von-allerliebst.de`).
- [ ] DB-Migration MSSQL → PostgreSQL. Achtung: **alle drei Konsumenten** der
      `dbdata` umbiegen — Contabo (39 Dateien mit hartkodierter IP),
      NAS-`bundesliga` (8), ASPX-`web.config` (1), dazu der Entwicklungsrechner.
- [ ] `bb_*`-Tabellen des Börsenbots mitnehmen. Nebeneffekt: das
      **400-MB-Hartlimit** ohne Autogrowth fällt weg, das am 05.08. den Handel
      stillgelegt hat.

---

## Phase 3 — Kündigen

- [ ] Erst wenn Phase 1 **und** 2 verifiziert sind: Centron kündigen.
- [ ] In der Kündigung ausdrücklich vermerken, dass die Domains **bereits
      transferiert** sind und keine Domain-Löschung beauftragt wird.
- [ ] Nach Vertragsende noch einmal beide Domains prüfen (NS, A, HTTPS).

**Notausgang**, falls die Frist doch knapp wird und ein Transfer nicht mehr
rechtzeitig durchgeht: bei der DENIC den Status **TRANSIT** setzen lassen. Die
Domain läuft dann direkt bei der DENIC weiter, statt gelöscht zu werden. Das ist
eine Rettungsleine, kein Plan.

---

## Geprüft und verworfen: das NAS als Webserver

Gemessen 05.08.2026, als die Frage aufkam, ob das Synology das Hosting
übernehmen könnte.

**Technisch läuft es schon:** `WebStation`, `Apache2.4`, `PHP7.3/7.4/8.0`,
`MariaDB10`, `phpMyAdmin`, `Joomla` sind installiert, Ports **80 und 443
lauschen bereits** (v4 und v6). Der Anschluss hat eine **öffentliche IPv4**
(89.182.200.26, kein CGNAT/DS-Lite) — eingehend wäre also möglich, `ddnsd`
läuft für DynDNS bereits.

**Trotzdem nein, aus drei gemessenen Gründen:**

1. **Die Kiste ist am Anschlag.** DS215j, ARMv7, 2 Kerne, 502 MB RAM — davon
   **36 MB frei**, und **577 MB Swap belegt**. Load **3,63 / 3,54 / 3,61** bei
   2 Kernen ≈ 180 % Dauerauslastung. Die CPU-Prozesse selbst sind harmlos
   (`syno-cloud-sync` 7 %), die Last ist also **I/O-gebunden** — dieselbe
   Ursache wie die 5 MB/s beim Backup-Erstlauf.
2. **Es ist die Backup-Senke.** Seit 03.08. liegt dort das komplette
   `/home/veit` inkl. `.ssh`, `config.py` und `.fred_key`
   ([[contabo-backup-auf-nas]]). Das NAS ins offene Internet zu stellen heißt,
   ausgerechnet die Kopie aller Secrets zu exponieren.
3. **Dort läuft der Live-Handel.** `signal_to_orders.py` und
   `random5_crossover_bot.py` mit `BOERSENBOT_LIVE=1`, dazu die
   Wetter-Rechner. Heimanschluss = Stromausfall, Router-Reboot und
   Zwangstrennung werden zu Handelsausfällen.

**Fazit:** Contabo (5 €/Monat) ist der falsche Ort zum Sparen — das Ziel bleiben
die 13 € bei Centron. Das NAS bleibt interne Maschine; für Fernzugriff ist
**Tailscale bereits installiert**, das braucht keine offene Portfreigabe.

## Reihenfolge in einem Satz

**Domains transferieren → Inhalte migrieren → kündigen.** Jede andere
Reihenfolge riskiert entweder eine tote `veitluther.de` oder eine verlorene
Domain von 2014.
