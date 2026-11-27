"""Frozen SEC filings, section boundaries and deterministic chunking."""

import gzip
import hashlib
import json
import re
from pathlib import Path

from lxml import html

ROOT = Path(__file__).resolve().parent
ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "Management’s Discussion and Analysis",
}
BOUNDARIES = {
    "Business": "1",
    "Risk Factors": "1A",
    "Unresolved Staff Comments": "1B",
    "Management’s Discussion and Analysis of Financial Condition and Results of Operations": "7",
    "Quantitative and Qualitative Disclosures About Market Risk": "7A",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sections(raw):
    tree = html.fromstring(raw)
    for node in tree.xpath(
        '//script|//style|//head|//*[local-name()="ix:hidden"]|//*[@style]'
    ):
        style = re.sub(r"\s", "", node.get("style", "")).lower()
        if (
            node.tag in {"script", "style", "head", "ix:hidden"}
            or "display:none" in style
        ):
            node.drop_tree()
    targets = {}
    labels = {k.lower().replace("’", "'"): v for k, v in BOUNDARIES.items()}
    for link in tree.xpath('//a[starts-with(@href,"#")]'):
        label = " ".join(link.text_content().split()).lower().replace("’", "'")
        if label in labels:
            targets.setdefault(labels[label], link.get("href")[1:])
        if len(targets) == len(labels):
            break
    if len(targets) != len(labels):
        raise ValueError("Incomplete SEC table of contents")
    for item, target in targets.items():
        nodes = tree.xpath("//*[@id=$target or @name=$target]", target=target)
        if not nodes:
            raise ValueError("Missing SEC anchor")
        nodes[0].text = f" FINPULSESECTION{item}END " + (nodes[0].text or "")
    for node in tree.xpath("//p|//div|//tr|//td|//br|//h1|//h2|//h3|//li"):
        node.tail = " " + (node.tail or "")
    text = re.sub(r"\s+", " ", tree.text_content())
    result = {}
    for item, end in [("1", "1A"), ("1A", "1B"), ("7", "7A")]:
        body = text.split("FINPULSESECTION" + item + "END ", 1)[1].split(
            " FINPULSESECTION" + end + "END", 1
        )[0]
        body = re.sub(
            r"Apple Inc\. \| 2024 Form 10-K \| \d+|\b\d+ PART (?:I|II) Item [0-9A, ]+(?= [A-Za-z])",
            " ",
            body,
        )
        result[item] = " ".join(body.split())
    return result


def chunks(meta, body, item, size=145, overlap=32):
    if size < 32 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk settings")
    words = body.split()
    for start in range(0, len(words), size - overlap):
        snippet = " ".join(words[start : start + size])
        key = f"{meta['accession']}|{meta['sha256']}|{item}|{start}|{snippet}"
        yield dict(
            meta, id=sha(key.encode())[:32], section=item, start=start, text=snippet
        )
        if start + size >= len(words):
            break


def corpus(folder=ROOT / "filings", size=145, overlap=32):
    records = {}
    for meta in json.loads((folder / "manifest.json").read_text()):
        raw = gzip.decompress((folder / (meta["ticker"] + ".html.gz")).read_bytes())
        if sha(raw) != meta["sha256"]:
            raise ValueError("Filing checksum mismatch")
        for item, body in sections(raw).items():
            for row in chunks(meta, body, item, size, overlap):
                records.setdefault(row["id"], row)
    return list(records.values())
