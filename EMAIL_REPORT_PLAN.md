# Email-Report-System: Plan für Implementierung heute Abend

## Anforderung
- **Trigger:** Täglich 06:00 UTC nach US-Börsentag
- **Inhalt:**
  1. Alle Transaktionen seit letztem Report
  2. Alle aktiven Positionen (current holdings)
  3. Anzahl verkaufter/liquidierter Positionen
  4. Aggregierter Saldo (net P&L)

---

## Architektur

### 1. Alpaca API Integration
**Endpoints zu nutzen:**
- `GET /v2/account` — Account-Info (Cash, Portfolio Value)
- `GET /v2/positions` — Aktuelle Positionen (qty, avg_fill_price, current_price)
- `GET /v2/orders?status=closed&limit=500` — Trade History (filled orders)
- `GET /v2/orders?status=open` — Offene Orders (optional für Report)

**Credentials (bereits in config.py):**
```
api_key: PK7C52Q5VZXZ5DDOIDCEEY7CKD
secret_key: BqgRvkRyUeanetTS8AEzvnTp5GMaPHcuWqFkCDjTgyLa
base_url: https://paper-api.alpaca.markets
```

### 2. Datenfluss
```
06:00 UTC → Cron startet python email_report_generator.py
            ↓
         Alpaca API queries
            ↓
         Transaktions-Aggregation (seit letztem Report)
            ↓
         HTML-Report generieren
            ↓
         Email versenden (veit.luther@gmx.de)
```

### 3. State Management
**Problem:** Wie wissen wir, welche Transaktionen "seit letztem Report" sind?

**Lösung 1 (einfach):** 
- Alpaca's `updated_at` Feld nutzen
- Nur Orders, deren `filled_at` > letzter Report-Zeitpunkt

**Lösung 2 (robust):**
- Lokal Timestamp speichern: `.last_report.txt` mit ISO-Datetime
- Bei nächstem Run: `orders?start_time=LAST_TIMESTAMP`

→ **Empfehlung:** Lösung 2, speichern in NAS `/home/veit/boersenbot/.last_report_time`

---

## Implementierungs-Checkliste

### Phase 1: Email-Report-Generator (Python)
**Datei:** `/home/veit/boersenbot/email_report_generator.py`

Funktion 1: `fetch_alpaca_data()`
- GET `/v2/account` → Cash, Portfolio Value, buying_power
- GET `/v2/positions` → Aktuelle Holdings (qty, symbol, avg_fill_price)
- GET `/v2/orders?status=closed` → gefüllte Orders seit last_report_time
- GET `/v2/orders?status=open` → offene Orders (optional)

Funktion 2: `aggregate_transactions(orders, last_report_time)`
- Filter: orders.filled_at > last_report_time
- Gruppierung: BUY vs SELL
- Summen: Qty, Value

Funktion 3: `generate_html_report(account, positions, transactions)`
- Account Summary (Cash, Portfolio Value, P&L)
- Transaction Table (seit letztem Report)
- Position Table (aktiv, qty, entry price, current price, unrealized P&L)
- Closed Positions Count

Funktion 4: `send_email(html_body, subject)`
- Ziel: veit.luther@gmx.de
- SMTP: depends on VPS mail setup (check config.py für Email-Creds)
- Format: HTML mit CSS-Styling

Funktion 5: `update_last_report_time()`
- Schreibe aktuellen UTC-Timestamp → `.last_report_time`

### Phase 2: Cron-Integration (NAS)
**Datei:** SSH in NAS, edit Crontab

```bash
# Täglich 06:00 UTC Email-Report
0 6 * * * cd /home/veit/boersenbot && python3 email_report_generator.py >> logs/email_report.log 2>&1
```

**Prerequisite:** NAS muss Mail-Client konfigurieren können (oder via VPS scp-trigger)

### Phase 3: Testing
1. Manuell ausführen: `python email_report_generator.py`
2. Email empfangen?
3. HTML-Rendering korrekt?
4. Transaktionen korrekt aggregiert?
5. Closed Positions Count korrekt?

---

## Fallstricke

1. **Timezone:** Alpaca API gibt Timestamps in UTC, muss konsistent bleiben
2. **Alpaca Market Hours:** 09:30-16:00 EST = 14:30-21:00 UTC
   - 06:00 UTC ist BEFORE market open (nachts in USA)
   - 06:00 UTC NACH Markt schliesst = "am nächsten Tag"
   - **Lösung:** Cron läuft 06:00 UTC = sammelt Orders vom Vortag (US-Zeit)
3. **Email SMTP:** VPS `config.py` muss Email-Creds haben (check ob SMTP-Service läuft)
4. **Alpaca API Rate Limit:** 200 requests/min — OK für Daily Report
5. **Closed vs Open Orders:** `status=closed` = erfüllte Orders, nicht "gestoppte"
   - "Closed" bei Alpaca = gefüllt ODER cancelled
   - Filter: `order.filled_at IS NOT NULL`

---

## SQL für Lokal-Tracking (optional)
Falls später DB-Sync nötig (z.B. für Audit):
```sql
CREATE TABLE alpaca_trades_log (
  id INT PRIMARY KEY AUTO_INCREMENT,
  order_id VARCHAR(100) UNIQUE,
  symbol VARCHAR(10),
  qty INT,
  filled_price DECIMAL(10,2),
  side ENUM('BUY', 'SELL'),
  filled_at DATETIME,
  created_at DATETIME DEFAULT NOW(),
  report_sent_at DATETIME
);
```

---

## Go/No-Go Checklist vor Deploy

- [ ] NAS: `email_report_generator.py` geschrieben
- [ ] Email-SMTP konfiguriert (veit.luther@gmx.de empfängt)
- [ ] Alpaca API Credentials verifiziert (Account aktiv: PA3Q0OSPULZE)
- [ ] `.last_report_time` speichern funktioniert
- [ ] Manueller Test erfolgreich (lokal auf Windows oder NAS)
- [ ] Cron-Job registriert (NAS, 06:00 UTC)
- [ ] Log-File Rotation (logs/email_report.log nicht zu groß)
- [ ] Error Handling (wenn Alpaca API ausfällt → Fehler-Email statt blank)

---

## Nächste Schritte (heute abend)

1. `email_report_generator.py` schreiben
2. Lokal testen (Windows oder SSH auf NAS)
3. Cron registrieren
4. Erste Email validieren
5. Alpaca Order-Placement integrieren (danach: signal → order translator)
