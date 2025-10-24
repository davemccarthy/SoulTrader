# SoulTrader - Clean Trading Algorithm

A simplified, database-driven trading analysis system built from lessons learned in the aiadvisor proof-of-concept project.

## Core Philosophy

- **Simplicity first** - Minimal moving parts
- **SQL-driven** - PostgreSQL for easy data inspection and querying
- **Clean separation** - Discovery, Analysis, Execution as distinct services
- **No feature creep** - Only what's needed for the core algorithm

## Core Algorithm

1. **Discovery** - Find stocks (Alpha Vantage, Yahoo Finance)
2. **Analysis** - Get advisor opinions, build consensus
3. **Execution** - Execute trades based on consensus threshold
4. **Tracking** - Monitor performance to identify patterns

## Project Structure

```
soultrader/
├── config/              # Django settings
├── core/                # Core trading logic
│   ├── models.py       # 8 simple models based on analysis.py
│   └── services/       # Business logic
│       ├── discovery.py    # Stock discovery
│       ├── analysis.py     # Consensus building
│       └── execution.py    # Trade execution
├── logs/               # Application logs
└── manage.py
```

## Setup

### 1. Create PostgreSQL Database

```bash
# Create database
createdb soultrader

# Or using psql:
psql postgres
CREATE DATABASE soultrader;
\q
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings:
# - Database credentials
# - API keys (as needed)
```

### 3. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run Migrations

```bash
# After you create your models in core/models.py:
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser

```bash
python manage.py createsuperuser
```

## Usage

### Run Smart Analysis

```bash
# Analyze specific user
python manage.py smartanalyse <username>

# Analyze all users
python manage.py smartanalyse --all

# Analyze holdings only (skip discovery)
python manage.py smartanalyse <username> --holdings-only

# Analyze discovery only (skip holdings)
python manage.py smartanalyse <username> --discovery-only
```

### Database Queries

Since this uses PostgreSQL, you can easily query data:

```sql
-- See all discoveries
SELECT * FROM core_discovery ORDER BY discovered_at DESC;

-- Check consensus for a session
SELECT stock_id, advisor_count, average_score 
FROM core_consensus 
WHERE sa_id = 123 
ORDER BY average_score DESC;

-- Find what we bought at the peak
SELECT d.stock_id, d.discovery_price, t.price as purchase_price
FROM core_discovery d
INNER JOIN core_trade t ON d.stock_id = t.stock_id
WHERE t.price >= d.discovery_price;
```

## Models (Overview)

Based on analysis.py design:

- **Stock** - Stock symbols and basic info
- **Advisor** - AI advisors (with discovery flag)
- **SmartAnalysis** - Analysis session (ties everything together)
- **Discovery** - Stocks discovered in a session
- **Recommendation** - Individual advisor opinions
- **Consensus** - Aggregated consensus (avg/total scores)
- **Holding** - Current positions (with volatile flag)
- **Trade** - Executed trades (linked to SA session)

## Key Features

### AnalysisFlags (BitFlags)
- `SA_HOLDINGS_ONLY` - Skip discovery
- `SA_DISCOVERY_ONLY` - Skip holdings
- `SA_REUSE_DATA` - Use cached data (future)

### Discovery Tracking
All discoveries are logged with:
- Discovery price
- Discovery time
- Source/method
- Used for pattern analysis

### Consensus Building
For each stock:
1. Get all advisor opinions
2. Calculate average confidence
3. Store in consensus table
4. Easy to query and analyze

## Development Workflow

1. **Design models** - Start with core/models.py
2. **Test in shell** - Use Django shell to test logic
3. **Build services** - Implement discovery, analysis, execution
4. **Add commands** - Management commands for running analysis
5. **Query & refine** - Use PostgreSQL to understand behavior
6. **Iterate** - Adjust based on real data

## Differences from aiadvisor Project

| aiadvisor (old) | soultrader (new) |
|-----------------|------------------|
| 18+ models | 8 simple models |
| 1,700+ line service | ~200 lines |
| SQLite | PostgreSQL |
| Complex ORM queries | Simple SQL |
| Retrofitted features | Built-in from start |
| Hard to debug | Easy to inspect |

## Next Steps

1. ✅ Project scaffolded
2. ⏳ Define models in `core/models.py`
3. ⏳ Run migrations
4. ⏳ Implement services
5. ⏳ Test in shell
6. ⏳ Add API/Portal (later)

## Notes

- Keep it simple - add complexity only when needed
- Use PostgreSQL for everything - it's your debugging tool
- SQL comments in code show intent
- Test each piece before moving on

---

Built with lessons learned from the aiadvisor playground project.

