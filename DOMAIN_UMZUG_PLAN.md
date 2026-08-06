# Domain-Umzug weg von Centron / internet1 — Action-Plan

**Stand:** 06.08.2026 (Phase 0 geschlossen, Centron hat geantwortet)
**Anlass:** Centron (13 €/Monat) soll perspektivisch fallen
(s. `BUNDESLIGA_MIGRATION_PLAN.md`). Vorher muss geklärt sein, was mit den
Domains passiert.

---

## Die Abarbeitungsreihenfolge — festgeschrieben 06.08.2026

Acht Schritte, jeder mit dem Ort, an dem er passiert. **Die Kündigung ist
beschlossen** (Betreiber, 06.08.) — sie ist aber **Schritt 8, nicht Schritt 1.**

> ⚠️ **Warum die Kündigung ganz hinten steht:** Das Paket Economy v2 enthält
> laut `BUNDESLIGA_MIGRATION_PLAN.md` **beides** — die ASPX-Seite (158.181.48.160)
> **und** die MSSQL `dbdata` (158.181.48.77). In `dbdata` liegen die **16
> `bb_`-Tabellen des Börsenbots**. Eine Kündigung vor Phase 2 nimmt damit nicht
> nur die Bundesliga-Seite mit, sondern die Datenbank, an der der **laufende
> Wetter-Handel** hängt (`bb_WeatherLadders`, Ladder-Logger, Autobuy, Settlement).

| # | Wo | Was |
|---|---|---|
| 1 | **INWX** | `kreativkommo.de` → DNS: `@` und `www` von `185.181.104.242` **auf 144.91.98.234 ändern** (bestehende Records bearbeiten). Panel-Testlauf. |
| 2 | **INWX** | Zonen für beide echten Domains **anlegen**, nichts transferieren. Records s. Umzugs-Karte unten, `stats.*` weglassen. |
| 3 | **Centron** | Kundencenter → **Network → Domains** → AuthCodes erzeugen. Erst jetzt — 30 Tage gültig. |
| 4 | **INWX** | KK-Antrag als *Providerwechsel*. **`frau-von-allerliebst.de` zuerst**, danach `veitluther.de`. |
| 5 | **INWX/VPS** | Verifizieren: NS, A-Record, HTTPS, `certbot renew --dry-run`. |
| — | | **Ab hier können die Domains nicht mehr verloren gehen.** |
| 6 | **VPS** | Bundesliga-ASPX → Flask (18 Seiten, rein lesend, Parallelbetrieb). |
| 7 | **VPS** | `dbdata` → PostgreSQL, **inkl. der 16 `bb_`-Tabellen**. Nebeneffekt: das 400-MB-Hartlimit fällt weg. |
| 8 | **Centron** | Billing → Vertrag ganz unten → **„Kündigung anfragen"**. Monatlich zum 12. Vermerk: Domains bereits transferiert, **keine** Domain-Löschung. Bestätigung ablegen. |

**Blocker Stand 06.08.2026:** Bei INWX ist **kein Login möglich** — Störung auf
deren Seite, nicht auf unserer. Damit hängen die Schritte 1 und 2. Kein
Zeitdruck dadurch: Schritt 3 darf ohnehin erst danach kommen (30-Tage-Frist des
AuthCodes), und die Kündigung ist monatlich terminierbar. **Wiedervorlage: Login
erneut versuchen.** Falls die Störung anhält, ist das ein Argument, den
Registrar noch einmal zu prüfen — aber erst, wenn sie mehr als ein paar Tage
dauert; eine einmalige Panel-Störung ist kein Auswahlkriterium.

---

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

## Phase 0 — GESCHLOSSEN 06.08.2026 durch Centrons Antwort

**Erledigt 05.08.2026:** Beide Domains sind laut Kundenpanel bei Centron
hinterlegt. Eine **Kündigungsfrist ist im Panel nicht erkennbar** — das heißt
nicht, dass es keine gibt.

### Antwort Centron, 06.08.2026 08:18 (Ticket #10963053, Billing/Head of Finance)

Drei Aussagen, wörtlich sinngemäß:

1. **„Die AuthCodes können Sie selbst im Kundencenter unter Network → Domains
   erstellen."** ⇒ Der AuthInfo-Blocker aus Phase 1 ist weg, **ohne Wartezeit**.
2. **„Das Paket Economy v2 können Sie immer zum 12. des Monats kündigen."**
   Weg dorthin: Billing → ganz unten der Vertrag → *Kündigung anfragen*, danach
   kommt eine Kündigungsbestätigung.
3. **„Die Domains haben eine jährliche Vertragslaufzeit und haben sich am 17.07.
   wieder um ein Jahr verlängert."**

**Was das für den Plan heißt:**

- **Die kritische Zahl des Plans ist erledigt.** Die AGB-Staffel „12 Monate ⇒
  3 Monate Frist" gilt für das Hosting-Paket **nicht** — es ist monatlich zum
  12. kündbar. Damit entfällt jeder Fristendruck, und die Frage
  Verbraucher- vs. Geschäftskundenvertrag (§ 312k) muss nicht mehr geklärt
  werden. Sie wurde auch nicht beantwortet — braucht es nicht mehr.
- **Die Domains sind KEINE Inklusivleistung**, sondern ein **eigener
  Jahresvertrag** — damit ist der zweite offene Punkt aus Phase 0 beantwortet.
  Laufzeit jetzt **17.07.2026 → 17.07.2027**.
- **Der Transfer kostet rund 11 bezahlte Monate.** Ein KK ist jederzeit möglich
  (DENIC bindet den Providerwechsel nicht an die Laufzeit beim abgebenden
  Registrar), das bereits gezahlte Domainjahr wird aber üblicherweise **nicht
  erstattet**, und INWX berechnet ab Transfer ein neues Jahr.
  **Trotzdem jetzt transferieren:** es geht um ~12 € gegen das Risiko einer
  Domain von 2014. Die Alternative — bis Juli 2027 warten und erst dann
  wechseln — ist genau die Reihenfolge, die dieser Plan ablehnt, und sie würde
  die Kündigung des Hostings elf Monate blockieren.
- **Was weiter unbeantwortet ist:** ob die Domainverträge bei Kündigung des
  Hosting-Pakets mitsterben. Muss nicht geklärt werden, solange die Domains
  **vor** der Kündigung weg sind — genau das ist die Reihenfolge des Plans.

⚠️ **AuthInfo nicht auf Vorrat ziehen.** Der DENIC-AuthInfo1 ist **30 Tage
gültig**. Erst ziehen, wenn die INWX-Zonen stehen und der KK-Antrag direkt
danach gestellt wird.

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

- [x] **Vertragsende + Frist schriftlich erfragen** — **raus am 05.08.2026,
      22:29**, von `veit.luther@gmx.de`, erfasst als **Ticket #10963053**,
      Kundennummer 326773. Fragt in einem Zug: (1) Laufzeit/Ende/Frist,
      (2) Verbraucher- oder Geschäftskundenvertrag samt § 312k-Schaltfläche,
      (3) AuthInfo-Codes für beide Domains. Ausdrücklich **keine** Kündigung.
      ~~Erbetene Antwort bis 20.08.2026 ⇒ Wiedervorlage.~~ **Antwort kam schon
      am 06.08.2026 08:18** — Wiedervorlage entfällt, s. Abschnitt oben.
- [x] Prüfen, ob die Domains **Inklusivleistung** des Hosting-Pakets sind.
      **Beantwortet 06.08.: nein** — eigener Jahresvertrag, verlängert bis
      17.07.2027. Ein Wegzug der Domains senkt den Paketpreis also nicht; er
      entkoppelt das Risiko. Das ist trotzdem richtig so.

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

- [x] `kreativkommo.de` bei INWX **registriert** — 05.08.2026, 22:45:30.
      DENIC-Status `active`, delegiert an `ns.inwx.de`, `ns2.inwx.de`,
      `ns3.inwx.eu`. **Offen:** `@` und `www` stehen noch auf der
      INWX-Parkseite `185.181.104.242` (RIPE: `INTERNETWORX-INFRA-2`) und
      müssen auf 144.91.98.234 **geändert** werden — die vorhandenen Records
      bearbeiten, keine neuen anlegen. Danach nginx-vhost + certbot auf dem VPS.
- [ ] **Nicht bei Centron registrieren** — sonst liegen drei statt zwei Domains
      in dem Vertrag, der weg soll.

> **60-Tage-Sperre beachten:** Eine frisch registrierte oder gerade
> transferierte Domain ist **60 Tage lang nicht transferierbar** (Registry-
> Standard, gilt bei jedem Anbieter). `kreativkommo.de` muss deshalb **sofort
> beim Zielregistrar** entstehen — wird sie „schnell mal woanders" registriert,
> ist sie zwei Monate festgenagelt. Für `veitluther.de` und
> `frau-von-allerliebst.de` ist das unkritisch (seit 2014 unverändert).
- [ ] Im INWX-DNS testweise einen A-Record auf 144.91.98.234 setzen und die
      Auflösung prüfen. Damit ist das Panel erprobt, bevor es ernst wird.
- [ ] Offen: wofür `kreativkommo.de` verwendet wird und ob sie dauerhaft auf den
      VPS zeigt (dann nginx-`server_name` + eigenes Zertifikat nötig).

### 1b — die beiden echten Domains nachziehen

- [x] ~~TTLs senken~~ — **entfällt.** Das Centron-Panel zeigt TTL **1800 s**
      (30 min), das ist niedrig genug. Schlimmstenfalls dauert die Umstellung
      eine halbe Stunde.
- [ ] **Zonen bei INWX vorab anlegen** — die vier A-Records aus der Tabelle
      oben, `stats.*` bewusst weglassen. Noch nicht aktivieren.
- [ ] **AuthInfo-Code** für beide Domains **selbst erzeugen**: Centron-
      Kundencenter → **Network → Domains** (so die Auskunft vom 06.08.). Kein
      Ticket, keine Wartezeit. **Erst unmittelbar vor dem KK-Antrag ziehen** —
      der Code ist 30 Tage gültig.
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

## Umzugs-Karte — alles, was beim neuen Registrar eingetippt wird

Zum Abarbeiten während des Umzugs. Nichts hiervon muss noch irgendwo
nachgeschlagen werden.

### Zone `veitluther.de` (LIVE — panel-bestätigt 05.08.2026)

```
@      A   144.91.98.234   TTL 1800
www    A   144.91.98.234   TTL 1800
```
`stats` **nicht** übernehmen. Kein MX, kein TXT, kein CAA, kein SRV.

### Zone `frau-von-allerliebst.de` (panel-bestätigt 05.08.2026)

```
@      A   158.181.48.160  TTL 1800     <- Centron-IIS, bis Phase 2 fertig ist
www    A   158.181.48.160  TTL 1800
```
`stats` **nicht** übernehmen. Nach der ASPX-Migration beide auf 144.91.98.234.

### Zone `kreativkommo.de` (neu)

Noch offen, wofür sie steht. Für den Testlauf genügt ein A-Record auf
144.91.98.234 — dann muss aber auch nginx davon wissen (s. u.).

### Was der Registrar sonst abfragt

| Feld | Wert |
|---|---|
| Domaininhaber (Halter) | eigene Adressdaten — müssen zustellbar sein, DENIC prüft |
| Admin-C | dieselbe Person; bei `.de` mit Wohnsitz in DE unproblematisch |
| AuthInfo | selbst zu erzeugen: Centron-Kundencenter → Network → Domains (30 Tage gültig) |
| Nameserver | die von INWX vorgegebenen; NICHT `ns*.internet1.de` eintragen |

### Nach dem Umzug auf dem VPS

```bash
ssh -i ~/.ssh/boersenbot_key veit@144.91.98.234
sudo certbot renew --dry-run          # muss durchlaufen
```
Erst wenn eine Domain **neu** auf den VPS zeigt, kommt ein `server_name` in
nginx dazu plus `sudo certbot --nginx -d <domain> -d www.<domain>`.
Aktuell kennt nginx nur `veitluther.de www.veitluther.de`.

### Prüfbefehle (nach jedem Schritt)

```powershell
Resolve-DnsName veitluther.de -Type NS -Server 8.8.8.8   # neue NS da?
Resolve-DnsName veitluther.de -Type A  -Server 8.8.8.8   # 144.91.98.234?
```
```bash
curl -sI https://veitluther.de | head -1
curl -sI https://frau-von-allerliebst.de/bundesliga/ | head -1
```

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

**Weg und Termin stehen seit 06.08. fest:** Kundencenter → **Billing** → ganz
unten der Vertrag → **„Kündigung anfragen"**. Kündbar **immer zum 12. des
Monats**, danach kommt eine Kündigungsbestätigung. Kein Fristendruck, keine
drei Monate — die AGB-Staffel greift für dieses Paket nicht.

- [ ] Erst wenn Phase 1 **und** 2 verifiziert sind: Centron kündigen.
- [ ] In der Kündigung ausdrücklich vermerken, dass die Domains **bereits
      transferiert** sind und keine Domain-Löschung beauftragt wird.
- [ ] **Kündigungsbestätigung abwarten und ablegen** — sie ist der einzige
      Beleg; das Panel zeigt keine Frist an.
- [ ] Nach Vertragsende noch einmal beide Domains prüfen (NS, A, HTTPS).

**Notausgang**, falls ein Transfer wider Erwarten nicht rechtzeitig durchgeht:
bei der DENIC den Status **TRANSIT** setzen lassen. Die Domain läuft dann direkt
bei der DENIC weiter, statt gelöscht zu werden. Das ist eine Rettungsleine, kein
Plan — und nach der Auskunft vom 06.08. sehr unwahrscheinlich nötig, weil die
Kündigung monatlich terminierbar ist und damit dem Transfer folgen kann statt
ihn zu treiben.

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
