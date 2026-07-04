"""
ETF holdings intelligence — fetch issuer snapshots, diff constituents, cross-fund lookup.

Snapshots live under ``.etf_holdings/{ETF}/{holdings_date}.json`` at project root.
Any advisor may import this module; the ``Etf`` advisor trades on daily diffs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

IGNORE_TICKERS: frozenset[str] = frozenset(
    {
        "CASH",
        "CASH&OTHER",
        "CASH OTHER",
        "FGXXX",
        "EXPIRY",
        "EUR",
        "JPY",
        "$EUR",
        "$JPY",
    }
)

THEME_LABELS: Dict[str, str] = {
    "aerospace_defense": "Aerospace & defense",
    "robotics_ai": "Robotics & AI",
    "ai_technology": "AI technology",
    "uranium_nuclear": "Uranium & nuclear",
    "semiconductors": "Semiconductors",
    "cybersecurity": "Cybersecurity",
    "cloud_software": "Cloud software",
    "biotech": "Biotech",
    "quantum": "Quantum",
    "disruptive_innovation": "Disruptive innovation",
    "genomics": "Genomics",
    "next_generation_internet": "Next generation internet",
    "autonomous_robotics": "Autonomous & robotics",
}


@dataclass(frozen=True)
class ETFConfig:
    etf: str
    theme: str
    provider: str
    url: str
    management_style: str  # passive | active
    issuer: str


@dataclass
class Holding:
    ticker: str
    name: str = ""
    cusip: str = ""
    weight_pct: Optional[float] = None
    shares: Optional[float] = None
    raw: Dict[str, Any] | None = None


@dataclass
class Snapshot:
    etf: str
    theme: str
    provider: str
    source_url: str
    holdings_date: str
    fetched_at_utc: str
    holdings: List[Holding]
    management_style: str = "passive"
    issuer: str = ""


@dataclass
class DiffResult:
    etf: str
    holdings_date: str
    previous_date: Optional[str]
    added: List[Holding] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)


@dataclass
class FundMembership:
    etf: str
    theme: str
    weight_pct: Optional[float]
    holdings_date: str
    management_style: str
    issuer: str
    provider: str
    name: str = ""


@dataclass
class EntrantEvent:
    symbol: str
    etf: str
    theme: str
    name: str
    weight_pct: Optional[float]
    holdings_date: str
    management_style: str
    issuer: str
    inclusion_type: str  # index_rebalance | new_holding


@dataclass
class RefreshResult:
    etf: str
    ok: bool
    error: Optional[str] = None
    snapshot_path: Optional[Path] = None
    holdings_date: Optional[str] = None
    diff: Optional[DiffResult] = None


DEFAULT_ETFS: Dict[str, ETFConfig] = {
    "QTUM": ETFConfig(
        etf="QTUM",
        theme="quantum",
        provider="defiance",
        url="https://www.defianceetfs.com/qtum-full-holdings/",
        management_style="passive",
        issuer="defiance",
    ),
    "BOTZ": ETFConfig(
        etf="BOTZ",
        theme="robotics_ai",
        provider="globalx",
        url="https://www.globalxetfs.com/funds/botz/",
        management_style="passive",
        issuer="globalx",
    ),
    "AIQ": ETFConfig(
        etf="AIQ",
        theme="ai_technology",
        provider="globalx",
        url="https://www.globalxetfs.com/funds/aiq/",
        management_style="passive",
        issuer="globalx",
    ),
    "URA": ETFConfig(
        etf="URA",
        theme="uranium_nuclear",
        provider="globalx",
        url="https://www.globalxetfs.com/funds/ura/",
        management_style="passive",
        issuer="globalx",
    ),
    "SMH": ETFConfig(
        etf="SMH",
        theme="semiconductors",
        provider="frenzycap",
        url="https://www.frenzycap.com/quote/SMH",
        management_style="passive",
        issuer="vaneck",
    ),
    "CIBR": ETFConfig(
        etf="CIBR",
        theme="cybersecurity",
        provider="firsttrust",
        url="https://www.ftportfolios.com/Retail/Etf/EtfHoldings.aspx?Ticker=CIBR",
        management_style="passive",
        issuer="firsttrust",
    ),
    "SKYY": ETFConfig(
        etf="SKYY",
        theme="cloud_software",
        provider="firsttrust",
        url="https://www.ftportfolios.com/Retail/Etf/EtfHoldings.aspx?Ticker=SKYY",
        management_style="passive",
        issuer="firsttrust",
    ),
    "XBI": ETFConfig(
        etf="XBI",
        theme="biotech",
        provider="spdr_xlsx",
        url="https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx",
        management_style="passive",
        issuer="spdr",
    ),
    "XAR": ETFConfig(
        etf="XAR",
        theme="aerospace_defense",
        provider="spdr_xlsx",
        url="https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xar.xlsx",
        management_style="passive",
        issuer="spdr",
    ),
    "ARKK": ETFConfig(
        etf="ARKK",
        theme="disruptive_innovation",
        provider="ark_csv",
        url="https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
        management_style="active",
        issuer="ark",
    ),
    "ARKG": ETFConfig(
        etf="ARKG",
        theme="genomics",
        provider="ark_csv",
        url="https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
        management_style="active",
        issuer="ark",
    ),
    "ARKW": ETFConfig(
        etf="ARKW",
        theme="next_generation_internet",
        provider="ark_csv",
        url="https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
        management_style="active",
        issuer="ark",
    ),
    "ARKQ": ETFConfig(
        etf="ARKQ",
        theme="autonomous_robotics",
        provider="ark_csv",
        url="https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_AUTONOMOUS_TECH._%26_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
        management_style="active",
        issuer="ark",
    ),
}

DEFAULT_ETF_LIST: tuple[str, ...] = tuple(DEFAULT_ETFS.keys())


class FirstTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._current_row: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in {"td", "th"} and self._in_cell:
            value = _clean(" ".join(self._cell_parts))
            self._current_row.append(value)
            self._in_cell = False
            self._cell_parts = []
        elif tag_l == "tr" and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = []


def default_holdings_dir() -> Path:
    try:
        from django.conf import settings

        return Path(settings.BASE_DIR) / ".etf_holdings"
    except Exception:
        return Path(".etf_holdings")


def theme_label(theme: str) -> str:
    return THEME_LABELS.get(theme, theme.replace("_", " ").title())


def fund_character_sentence(theme: str, management_style: str) -> str:
    label = theme_label(theme)
    if management_style == "active":
        return f"{label} active managed fund"
    return f"{label} passive index fund"


def infer_inclusion_type(config: ETFConfig, added_count: int) -> str:
    if config.management_style == "active":
        return "new_holding"
    if added_count >= 3:
        return "index_rebalance"
    return "new_holding"


def inclusion_label(inclusion_type: str) -> str:
    if inclusion_type == "index_rebalance":
        return "Index rebalance"
    return "New holding"


def is_valid_equity_ticker(ticker: str) -> bool:
    sym = _clean(ticker).upper()
    if not sym or sym in IGNORE_TICKERS:
        return False
    if sym.startswith("$"):
        return False
    if "." in sym and not sym.endswith((".A", ".B")):
        # foreign listings like 6861 JP are ok; require at least one letter
        pass
    return bool(re.match(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$", sym))


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_float(value: Any) -> Optional[float]:
    text = _clean(value).replace(",", "").replace("$", "").replace("%", "")
    if not text or text in {"-", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_date(value: str) -> str:
    text = _clean(value)
    for fmt in ("%m/%d/%Y", "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return text


def _get(url: str) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; soultrader-etf-holdings/1.0)",
        "Accept": "text/html,text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


def _row_value(row: Dict[str, Any], candidates: Iterable[str]) -> str:
    by_lower = {str(k).strip().lower(): v for k, v in row.items()}
    for candidate in candidates:
        value = by_lower.get(candidate.lower())
        if value is not None and _clean(value):
            return _clean(value)
    return ""


def _holding_from_row(row: Dict[str, Any]) -> Optional[Holding]:
    ticker = _row_value(row, ["ticker", "symbol", "identifier", "Ticker"])
    if not ticker or not is_valid_equity_ticker(ticker):
        return None
    name = _row_value(row, ["name", "company", "holding name", "security name"])
    weight = _row_value(
        row,
        ["% of net assets", "etf weight", "weight (%)", "weight %", "weight", "weighting", "market value weight"],
    )
    shares = _row_value(row, ["shares", "shares held", "shares/par value", "shares / quantity"])
    cusip = _row_value(row, ["cusip"])
    return Holding(
        ticker=ticker.upper(),
        name=name,
        cusip=cusip,
        weight_pct=_parse_float(weight),
        shares=_parse_float(shares),
        raw={str(k): v for k, v in row.items()},
    )


def _parse_csv_holdings(text: str) -> tuple[str, List[Holding]]:
    lines = text.splitlines()
    holdings_date = ""
    for line in lines[:10]:
        match = re.search(r"(?:Data as of|as of)\s+([A-Za-z0-9/, -]+)", line, flags=re.I)
        if match:
            holdings_date = _normalize_date(match.group(1))
            break

    header_idx = None
    for idx, line in enumerate(lines):
        low = line.lower()
        if "ticker" in low and ("name" in low or "company" in low):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("could not locate CSV holdings header row")

    reader = csv.DictReader(lines[header_idx:])
    holdings: List[Holding] = []
    for row in reader:
        if not holdings_date:
            row_date = _row_value(row, ["date"])
            if row_date:
                holdings_date = _normalize_date(row_date)
        holding = _holding_from_row(row)
        if holding is not None:
            holdings.append(holding)

    if not holdings:
        raise ValueError("CSV parsed but produced no holdings")
    return holdings_date, holdings


def _snapshot_from_config(
    config: ETFConfig,
    source_url: str,
    holdings_date: str,
    holdings: List[Holding],
) -> Snapshot:
    holdings = sorted(
        holdings,
        key=lambda row: -1.0 if row.weight_pct is None else row.weight_pct,
        reverse=True,
    )
    return Snapshot(
        etf=config.etf,
        theme=config.theme,
        provider=config.provider,
        source_url=source_url,
        holdings_date=holdings_date,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        holdings=holdings,
        management_style=config.management_style,
        issuer=config.issuer,
    )


def fetch_direct_csv(config: ETFConfig) -> Snapshot:
    csv_resp = _get(config.url)
    holdings_date, holdings = _parse_csv_holdings(csv_resp.text)
    if not holdings_date:
        holdings_date = date.today().isoformat()
    return _snapshot_from_config(config, config.url, holdings_date, holdings)


def fetch_firsttrust(config: ETFConfig) -> Snapshot:
    resp = _get(config.url)
    html = resp.text

    date_match = re.search(r"Holdings of the Fund as of\s+([0-9/]+)", html, flags=re.I)
    holdings_date = _normalize_date(date_match.group(1)) if date_match else date.today().isoformat()

    parser = FirstTableParser()
    parser.feed(html)

    holdings: List[Holding] = []
    headers: Optional[List[str]] = None
    for row in parser.rows:
        if "Security Name" in row and "Identifier" in row:
            headers = row
            continue
        if headers is None or len(row) < len(headers):
            continue
        holding = _holding_from_row(dict(zip(headers, row)))
        if holding is not None:
            holdings.append(holding)

    if not holdings:
        raise ValueError("First Trust page parsed but produced no holdings")
    return _snapshot_from_config(config, config.url, holdings_date, holdings)


def _xlsx_rows(content: bytes) -> List[List[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared.append("".join(texts))

        sheet_names = [name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")]
        if not sheet_names:
            raise ValueError("XLSX has no worksheets")
        root = ET.fromstring(zf.read(sorted(sheet_names)[0]))

        rows: List[List[str]] = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values: List[str] = []
            for cell in row.findall("a:c", ns):
                value = ""
                if cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//a:t", ns))
                else:
                    node = cell.find("a:v", ns)
                    if node is not None and node.text is not None:
                        value = node.text
                        if cell.attrib.get("t") == "s":
                            value = shared[int(value)]
                values.append(_clean(value))
            rows.append(values)
        return rows


def fetch_spdr_xlsx(config: ETFConfig) -> Snapshot:
    resp = _get(config.url)
    rows = _xlsx_rows(resp.content)

    holdings_date = date.today().isoformat()
    for row in rows[:10]:
        if len(row) >= 2 and row[0].lower().startswith("holdings"):
            holdings_date = _normalize_date(row[1].replace("As of", "").strip())
            break

    holdings: List[Holding] = []
    headers: Optional[List[str]] = None
    for row in rows:
        if "Ticker" in row and "Name" in row and "Weight" in row:
            headers = row
            continue
        if headers is None or len(row) < len(headers):
            continue
        holding = _holding_from_row(dict(zip(headers, row)))
        if holding is not None:
            holdings.append(holding)

    if not holdings:
        raise ValueError("SPDR XLSX parsed but produced no holdings")
    return _snapshot_from_config(config, config.url, holdings_date, holdings)


def fetch_frenzycap(config: ETFConfig) -> Snapshot:
    resp = _get(config.url)
    html = resp.text

    date_match = re.search(r"as of\s+([0-9-]+)", html, flags=re.I)
    holdings_date = _normalize_date(date_match.group(1)) if date_match else date.today().isoformat()

    parser = FirstTableParser()
    parser.feed(html)

    holdings: List[Holding] = []
    headers: Optional[List[str]] = None
    for row in parser.rows:
        if row[:3] == ["Symbol", "Name", "Weight %"]:
            headers = row
            continue
        if headers is None or len(row) < len(headers):
            continue
        if row[0] in {"—", "#"}:
            if row[0] == "#":
                break
            continue
        holding = _holding_from_row(dict(zip(headers, row)))
        if holding is not None:
            holdings.append(holding)

    if not holdings:
        raise ValueError("FrenzyCap table parsed but produced no holdings")
    return _snapshot_from_config(config, config.url, holdings_date, holdings)


def fetch_defiance(config: ETFConfig) -> Snapshot:
    resp = _get(config.url)
    html = resp.text

    date_match = re.search(r"Data as of\s+([0-9/]+)", html, flags=re.I)
    holdings_date = _normalize_date(date_match.group(1)) if date_match else date.today().isoformat()

    parser = FirstTableParser()
    parser.feed(html)
    if len(parser.rows) < 2:
        raise ValueError("no holdings table found")

    headers = parser.rows[0]
    holdings: List[Holding] = []
    for cells in parser.rows[1:]:
        holding = _holding_from_row(dict(zip(headers, cells)))
        if holding is not None:
            holdings.append(holding)

    if not holdings:
        raise ValueError("Defiance table parsed but produced no holdings")
    return _snapshot_from_config(config, config.url, holdings_date, holdings)


def _discover_csv_url(page_url: str, html: str, etf: str) -> str:
    patterns = [
        rf'https?://[^"\']+/{etf.lower()}[_-]full-holdings[_-]\d{{8}}\.csv',
        rf'https?://[^"\']+{etf}[^"\']+\.csv',
        r'https?://[^"\']+\.csv',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.I)
        if match:
            return match.group(0).replace("\\/", "/")

    href_match = re.search(r'href=["\']([^"\']+\.csv[^"\']*)["\']', html, flags=re.I)
    if href_match:
        return urljoin(page_url, href_match.group(1))
    raise ValueError("no CSV link found on issuer page")


def fetch_csv_link_page(config: ETFConfig) -> Snapshot:
    page = _get(config.url)
    csv_url = _discover_csv_url(config.url, page.text, config.etf)
    csv_resp = _get(csv_url)
    holdings_date, holdings = _parse_csv_holdings(csv_resp.text)
    if not holdings_date:
        holdings_date = date.today().isoformat()
    return _snapshot_from_config(config, csv_url, holdings_date, holdings)


def fetch_snapshot(config: ETFConfig) -> Snapshot:
    if config.provider == "defiance":
        return fetch_defiance(config)
    if config.provider == "ark_csv":
        return fetch_direct_csv(config)
    if config.provider == "firsttrust":
        return fetch_firsttrust(config)
    if config.provider == "spdr_xlsx":
        return fetch_spdr_xlsx(config)
    if config.provider == "frenzycap":
        return fetch_frenzycap(config)
    if config.provider in {"globalx", "ark"}:
        return fetch_csv_link_page(config)
    raise ValueError(f"unsupported provider {config.provider!r}")


def _snapshot_path(out_dir: Path, snapshot: Snapshot) -> Path:
    return out_dir / snapshot.etf / f"{snapshot.holdings_date}.json"


def snapshot_to_dict(snapshot: Snapshot) -> Dict[str, Any]:
    payload = asdict(snapshot)
    return payload


def snapshot_from_dict(payload: Dict[str, Any]) -> Snapshot:
    config = DEFAULT_ETFS.get(payload.get("etf", ""))
    holdings = [
        Holding(
            ticker=_clean(h.get("ticker")).upper(),
            name=h.get("name", ""),
            cusip=h.get("cusip", ""),
            weight_pct=h.get("weight_pct"),
            shares=h.get("shares"),
            raw=h.get("raw"),
        )
        for h in payload.get("holdings", [])
    ]
    return Snapshot(
        etf=payload["etf"],
        theme=payload.get("theme", config.theme if config else ""),
        provider=payload.get("provider", config.provider if config else ""),
        source_url=payload.get("source_url", ""),
        holdings_date=payload.get("holdings_date", ""),
        fetched_at_utc=payload.get("fetched_at_utc", ""),
        holdings=holdings,
        management_style=payload.get(
            "management_style", config.management_style if config else "passive"
        ),
        issuer=payload.get("issuer", config.issuer if config else ""),
    )


def load_snapshot(path: Path) -> Snapshot:
    return snapshot_from_dict(json.loads(path.read_text()))


def write_snapshot(out_dir: Path, snapshot: Snapshot, *, refresh: bool = False) -> Path:
    path = _snapshot_path(out_dir, snapshot)
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot_to_dict(snapshot), indent=2, sort_keys=True) + "\n")
    return path


def previous_snapshot_path(out_dir: Path, etf: str, holdings_date: str) -> Optional[Path]:
    etf_dir = out_dir / etf
    if not etf_dir.exists():
        return None
    paths = sorted(p for p in etf_dir.glob("*.json") if p.stem < holdings_date)
    return paths[-1] if paths else None


def latest_snapshot_path(out_dir: Path, etf: str) -> Optional[Path]:
    etf_dir = out_dir / etf
    if not etf_dir.exists():
        return None
    paths = sorted(etf_dir.glob("*.json"))
    return paths[-1] if paths else None


def diff_snapshots(current: Snapshot, previous: Snapshot) -> DiffResult:
    current_map = {h.ticker.upper(): h for h in current.holdings if h.ticker}
    previous_syms = {h.ticker.upper() for h in previous.holdings if h.ticker}
    added = sorted(
        (current_map[sym] for sym in current_map.keys() - previous_syms),
        key=lambda h: h.ticker,
    )
    removed = sorted(previous_syms - current_map.keys())
    return DiffResult(
        etf=current.etf,
        holdings_date=current.holdings_date,
        previous_date=previous.holdings_date,
        added=added,
        removed=removed,
    )


def diff_latest(out_dir: Path, etf: str) -> Optional[DiffResult]:
    path = latest_snapshot_path(out_dir, etf)
    if path is None:
        return None
    current = load_snapshot(path)
    prev_path = previous_snapshot_path(out_dir, etf, current.holdings_date)
    if prev_path is None:
        return DiffResult(
            etf=current.etf,
            holdings_date=current.holdings_date,
            previous_date=None,
            added=[],
            removed=[],
        )
    previous = load_snapshot(prev_path)
    return diff_snapshots(current, previous)


def refresh_snapshot(
    config: ETFConfig,
    out_dir: Path,
    *,
    refresh: bool = False,
) -> RefreshResult:
    try:
        snapshot = fetch_snapshot(config)
        prev_path = previous_snapshot_path(out_dir, config.etf, snapshot.holdings_date)
        path = write_snapshot(out_dir, snapshot, refresh=refresh)
        diff: Optional[DiffResult] = None
        if prev_path is not None:
            previous = load_snapshot(prev_path)
            diff = diff_snapshots(snapshot, previous)
        else:
            diff = DiffResult(
                etf=snapshot.etf,
                holdings_date=snapshot.holdings_date,
                previous_date=None,
                added=[],
                removed=[],
            )
        return RefreshResult(
            etf=config.etf,
            ok=True,
            snapshot_path=path,
            holdings_date=snapshot.holdings_date,
            diff=diff,
        )
    except Exception as exc:
        logger.warning("ETF refresh failed %s: %s", config.etf, exc)
        return RefreshResult(etf=config.etf, ok=False, error=f"{type(exc).__name__}: {exc}")


def refresh_snapshots(
    etfs: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    *,
    refresh: bool = False,
) -> List[RefreshResult]:
    root = out_dir or default_holdings_dir()
    symbols = [e.strip().upper() for e in (etfs or DEFAULT_ETF_LIST) if e]
    results: List[RefreshResult] = []
    for etf in symbols:
        config = DEFAULT_ETFS.get(etf)
        if config is None:
            results.append(RefreshResult(etf=etf, ok=False, error="unsupported ETF"))
            continue
        results.append(refresh_snapshot(config, root, refresh=refresh))
    return results


def entrants_from_refresh(results: Iterable[RefreshResult]) -> List[EntrantEvent]:
    events: List[EntrantEvent] = []
    for result in results:
        if not result.ok or result.diff is None:
            continue
        config = DEFAULT_ETFS.get(result.etf)
        if config is None:
            continue
        added_count = len(result.diff.added)
        inclusion_type = infer_inclusion_type(config, added_count)
        for holding in result.diff.added:
            if not is_valid_equity_ticker(holding.ticker):
                continue
            events.append(
                EntrantEvent(
                    symbol=holding.ticker.upper(),
                    etf=result.etf,
                    theme=config.theme,
                    name=holding.name,
                    weight_pct=holding.weight_pct,
                    holdings_date=result.diff.holdings_date,
                    management_style=config.management_style,
                    issuer=config.issuer,
                    inclusion_type=inclusion_type,
                )
            )
    return events


def funds_for_symbol(
    symbol: str,
    out_dir: Optional[Path] = None,
    etfs: Optional[Iterable[str]] = None,
) -> List[FundMembership]:
    sym = symbol.strip().upper()
    root = out_dir or default_holdings_dir()
    memberships: List[FundMembership] = []
    for etf in etfs or DEFAULT_ETF_LIST:
        path = latest_snapshot_path(root, etf)
        if path is None:
            continue
        snapshot = load_snapshot(path)
        for holding in snapshot.holdings:
            if holding.ticker.upper() != sym:
                continue
            memberships.append(
                FundMembership(
                    etf=snapshot.etf,
                    theme=snapshot.theme,
                    weight_pct=holding.weight_pct,
                    holdings_date=snapshot.holdings_date,
                    management_style=snapshot.management_style,
                    issuer=snapshot.issuer,
                    provider=snapshot.provider,
                    name=holding.name,
                )
            )
            break
    return memberships


def first_seen_in_fund(
    symbol: str,
    etf: str,
    out_dir: Optional[Path] = None,
) -> Optional[str]:
    sym = symbol.strip().upper()
    etf_dir = (out_dir or default_holdings_dir()) / etf
    if not etf_dir.exists():
        return None
    for path in sorted(etf_dir.glob("*.json")):
        snapshot = load_snapshot(path)
        if any(h.ticker.upper() == sym for h in snapshot.holdings):
            return snapshot.holdings_date
    return None


def themes_for_symbol(symbol: str, out_dir: Optional[Path] = None) -> Set[str]:
    return {m.theme for m in funds_for_symbol(symbol, out_dir=out_dir)}
