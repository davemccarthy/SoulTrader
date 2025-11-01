# SoulTrader Setup Guide

## Quick Start

### 1. PostgreSQL Database Setup

```bash
# Option 1: Command line
createdb soultrader

# Option 2: Using psql
psql postgres
CREATE DATABASE soultrader;
\q

# Option 3: Using a GUI tool (pgAdmin, Postico, etc.)
# Create database named: soultrader
```

### 2. Environment Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your PostgreSQL credentials:
nano .env
```

Required settings in `.env`:
```
DB_NAME=soultrader
DB_USER=postgres
DB_PASSWORD=your-postgres-password
DB_HOST=localhost
DB_PORT=5432
```

### 3. Virtual Environment & Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Database Migrations

**After you create your models in `core/models.py`:**

```bash
# Create migrations
python manage.py makemigrations core

# Apply migrations
python manage.py migrate

# Verify in PostgreSQL
psql soultrader
\dt  # List tables
```

### 5. Create Admin User

```bash
python manage.py createsuperuser
```

### 6. Test Django Admin

```bash
# Run development server
python manage.py runserver

# Visit: http://localhost:8000/admin
```

### 7. Configure Advisors

Advisors are automatically registered when modules are imported. You need to configure API keys and endpoints via Django Admin or Django shell.

**Access**: `http://localhost:8000/admin/core/advisor/`

#### Advisor Configuration Details

**Yahoo Finance** (No API key needed)
- Uses free `yfinance` library
- ✅ Discovery active (value-based)
- ✅ Analysis active

**Alpha Vantage**
- Endpoint: `https://www.alphavantage.co/query`
- Get API Key: https://www.alphavantage.co/support/#api-key
- ✅ Analysis active, ❌ Discovery disabled

**Finnhub**
- Endpoint: `https://finnhub.io/api/v1`
- Get API Key: https://finnhub.io/ (free tier available)
- ✅ Analysis active, ❌ No discovery

**Financial Modeling Prep (FMP)**
- Endpoint: `https://financialmodelingprep.com/api/v3`
- Get API Key: https://financialmodelingprep.com/developer/docs/ (free tier available)
- ✅ Analysis active, ❌ No discovery

**Polygon.io**
- Endpoint: *(not used - uses SDK)*
- Get API Key: https://polygon.io/ (free tier available)
- Also set in `config/settings.py` as `POLYGON_API_KEY` (for Gemini discovery)
- ✅ Analysis active, ❌ Discovery disabled

**Google Gemini**
- Endpoint: *(can be empty or model name)*
- Get API Key: https://aistudio.google.com/app/apikey
- ❌ Analysis disabled, ✅ Discovery active (news filtering)

**User Advisor**
- No API key needed
- ✅ Discovery active (manual), ❌ No analysis

#### Quick Setup via Django Shell

**Current Database Configuration:**

```python
from core.models import Advisor

# Alpha Vantage
adv = Advisor.objects.get(python_class="Alpha")
adv.endpoint = "https://www.alphavantage.co/query"
adv.key = "G0C346PZOQVFHNIF"
adv.enabled = True
adv.save()

# FMP (Financial Modeling Prep)
adv = Advisor.objects.get(python_class="FMP")
adv.endpoint = "https://financialmodelingprep.com/api/v3"
adv.key = "ynOKFYxumAVxrcJAGhzuN4ZBjCtQcbR7"
adv.enabled = False  # Currently disabled
adv.save()

# Finnhub
adv = Advisor.objects.get(python_class="Finnhub")
adv.endpoint = "https://finnhub.io/api/v1"
adv.key = "d3ch5a9r01qu125b63s0d3ch5a9r01qu125b63sg"
adv.enabled = False  # Currently disabled
adv.save()

# Polygon.io
adv = Advisor.objects.get(python_class="Polygon")
adv.endpoint = "https://api.polygon.io"
adv.key = "MSVhtqDKV9HyMOdla5UunU2EFs53MweY"
adv.enabled = True
adv.save()
# Also update settings.py: POLYGON_API_KEY = "MSVhtqDKV9HyMOdla5UunU2EFs53MweY"

# Gemini
adv = Advisor.objects.get(python_class="Gemini")
adv.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
adv.key = "AIzaSyCiVvWptLpmCGrQeTr2BaPfYJY04Sb21cU"
adv.enabled = True
adv.save()

# Yahoo (no configuration needed)
# User (no configuration needed)
```

#### Current Advisor Status

| Advisor | Discovery | Analysis | API Key Needed |
|---------|-----------|----------|----------------|
| **Yahoo** | ✅ Active | ✅ Active | ❌ No |
| **Alpha** | ❌ Disabled | ✅ Active | ✅ Yes |
| **Finnhub** | ❌ None | ✅ Active | ✅ Yes |
| **FMP** | ❌ None | ✅ Active | ✅ Yes |
| **Polygon** | ❌ Disabled | ✅ Active | ✅ Yes |
| **Gemini** | ✅ Active | ❌ Disabled | ✅ Yes |
| **User** | ✅ Active | ❌ None | ❌ No |

#### Important Notes

1. **Polygon API Key**: Also stored in `config/settings.py` as `POLYGON_API_KEY` (used by Gemini discovery for news fetching)

2. **Advisor Registration**: Happens automatically when advisor modules are imported (via `register()` calls)

3. **Enabled Status**: Set `enabled=True` for advisors you want active

4. **Free Tier Limits**: All APIs have rate limits - be aware of call restrictions

## Development Workflow

### Phase 1: Models (Current)

1. Edit `core/models.py` based on your `analysis.py` classes
2. Keep SQL queries in comments to document intent
3. Run migrations
4. Test in Django shell

```bash
python manage.py shell

>>> from core.models import Stock, Advisor
>>> # Test your models
```

### Phase 2: Services

1. Implement `core/services/discovery.py`
2. Implement `core/services/analysis.py`
3. Implement `core/services/execution.py`
4. Test each service independently in shell

### Phase 3: Management Command

1. Implement `core/management/commands/smartanalyse.py`
2. Wire up the services
3. Test end-to-end

```bash
python manage.py smartanalyse testuser
```

### Phase 4: API/Portal (Later)

Only after core logic is proven:
```bash
python manage.py startapp api
python manage.py startapp portal
```

## Useful Commands

### Django Shell

```bash
# Interactive Python shell with Django loaded
python manage.py shell

# Execute inline command
python manage.py shell -c "from core.models import Stock; print(Stock.objects.count())"
```

### Database Inspection

```bash
# Connect to PostgreSQL
psql soultrader

# Useful queries
\dt                          # List tables
\d core_stock               # Describe table structure
SELECT * FROM core_stock;   # Query data
```

### Migrations

```bash
# Create migrations
python manage.py makemigrations

# Show SQL that will run
python manage.py sqlmigrate core 0001

# Apply migrations
python manage.py migrate

# Show migration status
python manage.py showmigrations
```

### Development Server

```bash
# Run server
python manage.py runserver

# Run on different port
python manage.py runserver 8080
```

## Next Steps

1. ✅ Project structure created
2. ⏳ **Define models** in `core/models.py` (based on analysis.py)
3. ⏳ Run migrations to create tables
4. ⏳ Register models in `core/admin.py` for Django admin
5. ⏳ Implement service functions
6. ⏳ Test in Django shell
7. ⏳ Build management command
8. ⏳ Run analysis and collect data

## Tips

- **Keep models.py simple** - Match your analysis.py classes
- **Use PostgreSQL client** - Query directly to understand data
- **Test incrementally** - One service at a time
- **SQL comments** - Document queries in code
- **Django admin** - Great for initial data inspection

## Reference

Your original design: `/Users/davidmccarthy/Development/scratch/analysis.py`
Old project (lessons learned): `/Users/davidmccarthy/Development/CursorAI/Django/aiadvisor`

