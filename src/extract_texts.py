from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
import pandas as pd
from src.utils import html_to_text

def extract_documents(archive_path: str, output_csv: str = "data/processed/documents.csv") -> pd.DataFrame:
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    rows = []
    with zipfile.ZipFile(archive, "r") as zf:
        html_files = [name for name in zf.namelist() if name.endswith(".html")]
        for name in html_files:
            if "/content/dam/" in name:
                continue
            raw_html = zf.read(name).decode("utf-8", errors="ignore")
            text = html_to_text(raw_html)
            if len(text) < 200:
                continue
            rows.append(
                {
                    "path": name,
                    "title": Path(name).stem,
                    "text": text,
                    "text_length": len(text),
                }
            )

    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Path to twitter-data-staticsite.zip")
    parser.add_argument("--output", default="data/processed/documents.csv")
    args = parser.parse_args()

    df = extract_documents(args.archive, args.output)
    print(f"Saved {len(df)} documents to {args.output}")


if __name__ == "__main__":
    main()
