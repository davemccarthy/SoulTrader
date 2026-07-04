# ETF Entrant Events

Manual event log for the ETF entrant experiment. These are raw constituent additions
detected by diffing local holdings snapshots.

## 2026-06-24 Poll

Detection run:
- Poll date: 2026-06-24
- Signal: stock present in latest ETF snapshot and absent from previous local snapshot
- Price-test anchor: use the first-seen holdings date unless later evidence shows a later publication date

### XBI - SPDR S&P Biotech ETF

- Theme: biotech
- Previous snapshot: 2026-06-17
- First-seen holdings date: 2026-06-23
- Source file: `.etf_holdings/XBI/2026-06-23.json`

| Symbol | Name | ETF weight |
| --- | --- | ---: |
| ACHV | ACHIEVE LIFE SCIENCES INC | 0.067799% |
| AVTX | AVALO THERAPEUTICS INC | 0.362764% |
| BBOT | BRIDGEBIO ONCOLOGY THERAPEUT | 0.061150% |
| CABA | CABALETTA BIO INC | 0.159137% |
| CADL | CANDEL THERAPEUTICS INC | 0.184248% |
| IMMX | IMMIX BIOPHARMA INC | 0.209035% |
| JBIO | JADE BIOSCIENCES INC | 0.337776% |
| MPLT | MAPLIGHT THERAPEUTICS INC | 0.143116% |
| OVID | OVID THERAPEUTICS INC | 0.114443% |
| SGMT | SAGIMET BIOSCIENCES INC A | 0.069583% |
| SLDB | SOLID BIOSCIENCES INC | 0.227552% |

Removed in same diff: `AKBA`, `AVXL`, `IVVD`, `RCKT`.

### XAR - SPDR S&P Aerospace & Defense ETF

- Theme: aerospace_defense
- Previous snapshot: 2026-06-17
- First-seen holdings date: 2026-06-23
- Source file: `.etf_holdings/XAR/2026-06-23.json`

| Symbol | Name | ETF weight |
| --- | --- | ---: |
| FLY | FIREFLY AEROSPACE INC | 0.910008% |
| PKE | PARK AEROSPACE CORP | 0.344032% |
| SATL | SATELLOGIC INC A | 0.483559% |
| SPCE | VIRGIN GALACTIC HOLDINGS INC | 0.290077% |
| VOYG | VOYAGER TECHNOLOGIES INC A | 0.378908% |
| YSS | YORK SPACE SYSTEMS INC | 0.472852% |

Removed in same diff: none.

### Non-events / Ignored

- `CIBR` removed `$EUR` and `$JPY`; treated as currency/cash artifacts, not stock removals.
- `QTUM`, `BOTZ`, `AIQ`, `URA`, `SMH`, `SKYY`, `ARKK`, `ARKG`, `ARKW`, and `ARKQ` had no stock additions/removals in this poll.

## 2026-06-30 Poll

Detection run:
- Poll date: 2026-06-30
- Signal: stock present in latest ETF snapshot and absent from previous local snapshot
- Price-test anchor: use the first-seen holdings date unless later evidence shows a later publication date

### QTUM - Defiance Quantum ETF

- Theme: quantum
- Previous snapshot: 2026-06-29
- First-seen holdings date: 2026-06-30
- Source file: `.etf_holdings/QTUM/2026-06-30.json`

| Symbol | Name | ETF weight |
| --- | --- | ---: |
| HONA | Honeywell Aerospace Inc | 0.590000% |

Removed in same diff: none.

### Non-events / Ignored

- `BOTZ`, `AIQ`, `URA`, `SMH`, `CIBR`, `SKYY`, `XBI`, `XAR`, `ARKK`, `ARKG`, `ARKW`, and `ARKQ` had no stock additions/removals in this poll.
