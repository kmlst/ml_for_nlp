"""Collect FOMC statements and minutes from official Federal Reserve pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep
from typing import Literal
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (
    FED_BASE_URL,
    FOMC_CALENDAR_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from .preprocessing import clean_text, count_words


DocumentType = Literal["statement", "minutes"]

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DATE_IN_URL_RE = re.compile(r"(?P<ymd>(?:19|20)\d{6})")
HISTORICAL_HEADING_RE = re.compile(
    r"(?P<date_text>[A-Za-z/]+\s+\d{1,2}(?:\s*-\s*(?:[A-Za-z]+\s*)?\d{1,2})?)\s+Meeting\s+-\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

RATE_DECISION_PATTERNS = {
    "hike": [
        r"\b(decided|voted|agreed|approved)\b.{0,80}\b(raise|raising|increase|increasing)\b.{0,80}\b(target|federal funds|discount rate)\b",
        r"\b(raise|raising|increase|increasing|increased)\b.{0,80}\b(target range|target for the federal funds rate|federal funds rate)\b",
    ],
    "cut": [
        r"\b(decided|voted|agreed|approved)\b.{0,80}\b(lower|lowering|reduce|reducing|decrease|decreasing|cut)\b.{0,80}\b(target|federal funds|discount rate)\b",
        r"\b(lower|lowering|reduce|reducing|decrease|decreasing|cut)\b.{0,80}\b(target range|target for the federal funds rate|federal funds rate)\b",
    ],
    "hold": [
        r"\b(decided|voted|agreed)\b.{0,80}\b(maintain|keep|leave)\b.{0,80}\b(target range|target for the federal funds rate|federal funds rate)\b",
        r"\b(will|would|shall|to)\b.{0,20}\b(maintain|keep|leave)\b.{0,80}\b(target range|target for the federal funds rate|federal funds rate)\b",
        r"\b(maintain|maintaining|keep|keeping|kept|leave|leaving)\b.{0,80}\b(target range|target for the federal funds rate|federal funds rate)\b",
        r"\b(target range|target for the federal funds rate|federal funds rate)\b.{0,80}\b(unchanged|maintained)\b",
    ],
}


@dataclass(frozen=True)
class MeetingLinks:
    """URLs attached to one FOMC meeting."""

    date: str
    year: int
    meeting_id: str
    statement_url: str | None = None
    minutes_url: str | None = None


def ensure_output_dirs() -> None:
    """Create expected data directories."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str, timeout: int = 30) -> str:
    """Fetch a Federal Reserve HTML page."""

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "ml-for-nlp-fomc-project/1.0"},
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def parse_date_from_url(url: str) -> str | None:
    """Extract YYYY-MM-DD from common Fed FOMC URLs."""

    match = DATE_IN_URL_RE.search(url)
    if not match:
        return None
    ymd = match.group("ymd")
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


def parse_meeting_date(date_text: str, year: int) -> str:
    """Parse Fed meeting date labels and use the final meeting day."""

    date_text = date_text.replace("*", "").strip()
    date_text = re.sub(r"\s+", " ", date_text)
    match = re.match(
        r"(?P<month1>[A-Za-z]+)(?:/(?P<slash_month2>[A-Za-z]+))?\s+"
        r"(?P<day1>\d{1,2})"
        r"(?:\s*-\s*(?:(?P<month2>[A-Za-z]+)\s*)?(?P<day2>\d{1,2}))?$",
        date_text,
    )
    if not match:
        raise ValueError(f"Cannot parse meeting date: {date_text!r}")

    month1 = MONTHS[match.group("month1").lower()]
    day1 = int(match.group("day1"))
    slash_month2_text = match.group("slash_month2")
    month2_text = match.group("month2")
    day2_text = match.group("day2")
    if month2_text:
        month = MONTHS[month2_text.lower()]
    elif slash_month2_text and day2_text and int(day2_text) < day1:
        month = MONTHS[slash_month2_text.lower()]
    else:
        month = month1
    day = int(day2_text) if day2_text else day1
    parsed_year = year
    if day2_text and month2_text is None and day < day1 and month1 == 12:
        parsed_year += 1
    return date(parsed_year, month, day).isoformat()


def _historical_year_url(year: int) -> str:
    return f"{FED_BASE_URL}/monetarypolicy/fomchistorical{year}.htm"


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_historical_year_page(year: int, html: str) -> list[MeetingLinks]:
    soup = BeautifulSoup(html, "html.parser")
    meetings: list[MeetingLinks] = []

    panels = soup.select("div.panel")
    for panel in panels:
        heading = panel.select_one(".panel-heading") or panel.find(["h3", "h4", "h5", "h6"])
        if heading is None:
            continue
        heading_text = _normalize_space(heading.get_text(" ", strip=True))
        match = HISTORICAL_HEADING_RE.search(heading_text)
        if not match:
            continue
        meeting_date = parse_meeting_date(match.group("date_text"), int(match.group("year")))
        row = {
            "date": meeting_date,
            "year": int(meeting_date[:4]),
            "meeting_id": meeting_date,
            "statement_url": None,
            "minutes_url": None,
        }
        for link in panel.find_all("a", href=True):
            label = _normalize_space(link.get_text(" ", strip=True)).lower()
            href = urljoin(FED_BASE_URL, link["href"])
            href_lower = href.lower()
            if label == "statement":
                row["statement_url"] = href
            is_minutes_link = (
                label == "minutes"
                or ("fomcminutes" in href_lower and href_lower.endswith(".htm"))
                or ("/fomc/minutes/" in href_lower and href_lower.endswith(".htm"))
                or bool(re.search(r"/monetarypolicy/fomc\d{8}\.htm$", href_lower))
            )
            link_date = parse_date_from_url(href)
            if is_minutes_link and (link_date is None or link_date == meeting_date):
                row["minutes_url"] = href
        meetings.append(MeetingLinks(**row))

    return meetings


def _parse_current_calendar_page(html: str, start_year: int, end_year: int) -> list[MeetingLinks]:
    """Parse the Fed calendar page, using date-bearing statement/minutes URLs."""

    soup = BeautifulSoup(html, "html.parser")
    by_date: dict[str, dict[str, str | int | None]] = {}
    for link in soup.find_all("a", href=True):
        href = urljoin(FED_BASE_URL, link["href"])
        url_lower = href.lower()
        label = _normalize_space(link.get_text(" ", strip=True)).lower()
        meeting_date = parse_date_from_url(href)
        if not meeting_date:
            continue
        year = int(meeting_date[:4])
        if year < start_year or year > end_year:
            continue

        document_type: DocumentType | None = None
        if "fomcminutes" in url_lower and label == "html":
            document_type = "minutes"
        elif "/pressreleases/monetary" in url_lower and label == "html":
            document_type = "statement"
        elif "fomcstatement" in url_lower and label == "html":
            document_type = "statement"
        if document_type is None:
            continue

        by_date.setdefault(
            meeting_date,
            {
                "date": meeting_date,
                "year": year,
                "meeting_id": meeting_date,
                "statement_url": None,
                "minutes_url": None,
            },
        )
        by_date[meeting_date][f"{document_type}_url"] = href

    return [MeetingLinks(**row) for row in by_date.values()]


def collect_fomc_meeting_links(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Collect official statement and minutes URLs for each meeting."""

    meetings: list[MeetingLinks] = []
    for year in range(start_year, min(end_year, 2020) + 1):
        html = fetch_html(_historical_year_url(year))
        meetings.extend(_parse_historical_year_page(year, html))
        sleep(0.1)

    if end_year >= 2021:
        html = fetch_html(FOMC_CALENDAR_URL)
        meetings.extend(_parse_current_calendar_page(html, max(start_year, 2021), end_year))

    df = pd.DataFrame([meeting.__dict__ for meeting in meetings])
    if df.empty:
        return pd.DataFrame(columns=["date", "year", "meeting_id", "statement_url", "minutes_url"])
    return (
        df.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def collect_fomc_statement_urls(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Return one row per meeting with a statement URL when available."""

    columns = ["date", "year", "meeting_id", "statement_url"]
    df = collect_fomc_meeting_links(start_year, end_year)
    return df.loc[df["statement_url"].notna(), columns].reset_index(drop=True)


def collect_fomc_minutes_urls(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Return one row per meeting with a minutes URL when available."""

    columns = ["date", "year", "meeting_id", "minutes_url"]
    df = collect_fomc_meeting_links(start_year, end_year)
    return df.loc[df["minutes_url"].notna(), columns].reset_index(drop=True)


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()

    selectors = [
        "div.article__content",
        "div#article",
        "article",
        "main",
        "body",
    ]
    candidates: list[str] = []
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            text = node.get_text(" ", strip=True)
            if len(text) > 100:
                candidates.append(text)
    if not candidates:
        candidates.append(soup.get_text(" ", strip=True))
    text = max(candidates, key=len)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_document_text(url: str) -> str:
    """Download and extract text from an official FOMC HTML document."""

    if not isinstance(url, str) or not url.strip():
        return ""
    if url.lower().endswith(".pdf"):
        raise ValueError(f"PDF extraction is not enabled for {url}; prefer Fed HTML links.")
    html = fetch_html(url)
    return _extract_main_text(html)


def infer_rate_decision(text: str | None) -> str:
    """Infer hike/cut/hold from explicit decision language."""

    cleaned = clean_text(text)
    if not cleaned:
        return "unknown"
    for label in ("hike", "cut", "hold"):
        for pattern in RATE_DECISION_PATTERNS[label]:
            if re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL):
                return label
    return "unknown"


def apply_manual_rate_decisions(
    df: pd.DataFrame,
    manual_path: Path = RAW_DATA_DIR / "rate_decisions_manual.csv",
) -> pd.DataFrame:
    """Override inferred labels with an optional hand-checked CSV."""

    if not manual_path.exists():
        return df
    manual = pd.read_csv(manual_path, dtype={"date": str, "rate_decision": str})
    if "date" not in manual or "rate_decision" not in manual:
        return df
    overrides = manual.dropna(subset=["date", "rate_decision"]).set_index("date")["rate_decision"]
    result = df.copy()
    result["rate_decision"] = result.apply(
        lambda row: overrides.get(row["date"], row.get("rate_decision", "unknown")),
        axis=1,
    )
    return result


def build_document_dataset(
    links: pd.DataFrame,
    document_type: DocumentType,
    sleep_seconds: float = 0.1,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Download, clean and count one FOMC corpus."""

    url_col = f"{document_type}_url"
    text_col = f"{document_type}_text"
    clean_col = f"{document_type}_clean_text"
    n_words_col = f"{document_type}_n_words"
    rows: list[dict[str, object]] = []

    for row in links.to_dict(orient="records"):
        output = dict(row)
        url = output.get(url_col)
        try:
            text = download_document_text(str(url))
            output[text_col] = text
            output[clean_col] = clean_text(text)
            output[n_words_col] = count_words(text)
            output[f"{document_type}_download_error"] = ""
        except Exception as exc:
            if not continue_on_error:
                raise
            output[text_col] = ""
            output[clean_col] = ""
            output[n_words_col] = 0
            output[f"{document_type}_download_error"] = str(exc)
        rows.append(output)
        sleep(sleep_seconds)

    result = pd.DataFrame(rows)
    if document_type == "statement":
        result["rate_decision"] = result[text_col].map(infer_rate_decision)
        result = apply_manual_rate_decisions(result)
    return result


def build_statements_dataset(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Collect and download the statements corpus."""

    ensure_output_dirs()
    links = collect_fomc_statement_urls(start_year, end_year)
    df = build_document_dataset(links, "statement")
    df.to_csv(PROCESSED_DATA_DIR / "fomc_statements.csv", index=False)
    return df


def build_minutes_dataset(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Collect and download the minutes corpus."""

    ensure_output_dirs()
    links = collect_fomc_minutes_urls(start_year, end_year)
    df = build_document_dataset(links, "minutes")
    df.to_csv(PROCESSED_DATA_DIR / "fomc_minutes.csv", index=False)
    return df


def merge_statements_minutes(
    statements: pd.DataFrame | None = None,
    minutes: pd.DataFrame | None = None,
    output_path: Path = PROCESSED_DATA_DIR / "fomc_merged.csv",
) -> pd.DataFrame:
    """Merge statements and minutes by meeting date."""

    ensure_output_dirs()
    if statements is None:
        statements = pd.read_csv(PROCESSED_DATA_DIR / "fomc_statements.csv", dtype={"date": str})
    if minutes is None:
        minutes = pd.read_csv(PROCESSED_DATA_DIR / "fomc_minutes.csv", dtype={"date": str})

    merged = statements.merge(
        minutes,
        on=["date", "year", "meeting_id"],
        how="outer",
        suffixes=("", ""),
    )
    if "rate_decision" not in merged:
        merged["rate_decision"] = merged.get("statement_text", "").map(infer_rate_decision)
    merged = apply_manual_rate_decisions(merged)

    ordered_columns = [
        "date",
        "year",
        "meeting_id",
        "statement_url",
        "minutes_url",
        "statement_text",
        "minutes_text",
        "statement_clean_text",
        "minutes_clean_text",
        "statement_n_words",
        "minutes_n_words",
        "rate_decision",
    ]
    remaining = [column for column in merged.columns if column not in ordered_columns]
    merged = merged[[column for column in ordered_columns if column in merged.columns] + remaining]
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(output_path, index=False)
    return merged
