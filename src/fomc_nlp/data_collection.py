"""Collect FOMC statements and minutes from official Federal Reserve pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from time import sleep
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (
    FED_BASE_URL,
    FOMC_CALENDAR_URL,
    RAW_DATA_DIR,
)


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

        document_type: str | None = None
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


def _download_or_error(
    url: str | float | None,
    continue_on_error: bool = True,
) -> tuple[str, str]:
    """Download one document and return text plus an error string."""

    if not isinstance(url, str) or not url.strip():
        return "", "missing_url"
    try:
        return download_document_text(url), ""
    except Exception as exc:
        if not continue_on_error:
            raise
        return "", str(exc)


def build_raw_documents_dataset(
    links: pd.DataFrame,
    sleep_seconds: float = 0.1,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Download statements and minutes without adding analysis columns."""

    rows: list[dict[str, object]] = []
    for row in links.to_dict(orient="records"):
        statement_text, statement_error = _download_or_error(
            row.get("statement_url"),
            continue_on_error=continue_on_error,
        )
        minutes_text, minutes_error = _download_or_error(
            row.get("minutes_url"),
            continue_on_error=continue_on_error,
        )
        rows.append(
            {
                "date": row["date"],
                "year": row["year"],
                "meeting_id": row["meeting_id"],
                "statement_url": row.get("statement_url"),
                "minutes_url": row.get("minutes_url"),
                "statement_text": statement_text,
                "minutes_text": minutes_text,
                "statement_download_error": statement_error,
                "minutes_download_error": minutes_error,
            }
        )
        sleep(sleep_seconds)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
