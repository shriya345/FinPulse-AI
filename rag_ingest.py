"""Refresh the three pinned SEC filings into a new, independently verified snapshot."""

import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import argparse

import httpx
from lxml import html
from rag_docs import ROOT, sections, sha


def fetch_snapshot(destination, identification):
    if not identification or len(identification.strip()) < 12:
        raise ValueError("Set SEC_USER_AGENT to your project and real contact")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty staging directory to preserve archived filings")
    manifest = json.loads((ROOT / "filings/manifest.json").read_text())
    with httpx.Client(
        proxy=os.getenv("HTTPS_PROXY") or None,
        headers={"User-Agent": identification},
        timeout=45,
    ) as client:

        def get(url):
            for retry in range(3):
                time.sleep(1 + retry)
                response = client.get(url)
                if response.status_code not in (429, 503) or retry == 2:
                    response.raise_for_status()
                    return response.content
                delay = response.headers.get("Retry-After", "2")
                if not delay.isdigit() or int(delay) > 60:
                    response.raise_for_status()
                time.sleep(int(delay))

        for m in manifest:
            detail = get(
                m["url"].rsplit("/", 1)[0] + "/" + m["accession"] + "-index.htm"
            )
            header = html.fromstring(detail).text_content()
            if not all(
                s in header for s in (m["filed"], m["period"], m["accession"], "10-K")
            ):
                raise ValueError("Filing metadata mismatch")
            raw = get(m["url"])
            sections(raw)
            if sha(raw) != m["sha256"]:
                raise ValueError(
                    "Archived SEC filing changed; review before reindexing"
                )
            (destination / (m["ticker"] + ".html.gz")).write_bytes(
                gzip.compress(raw, mtime=0)
            )
            m["retrieved"] = datetime.now(timezone.utc).isoformat()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    fetch_snapshot(parser.parse_args().destination, os.getenv("SEC_USER_AGENT"))
