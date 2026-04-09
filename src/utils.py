from __future__ import annotations

import re
from pathlib import Path
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return clean_text(text)


def slug_from_path(path: str) -> str:
    return Path(path).stem.lower().replace("_", "-")
