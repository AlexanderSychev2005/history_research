#!/usr/bin/env python3
"""PDF -> canonical text (the `texts/<doc>.json` of eval/README.md) for the two sample papers.

  python prepare/prepare_texts.py [--papers papers] [--out texts] [--only alford2026]

Needs pdfplumber (pyproject.toml; run it with `uv run python prepare/prepare_texts.py`). The two papers have different layouts, so what is known about each
(font sizes, margins) is in CONFIG; everything else is shared. A third paper needs a CONFIG entry.
The script prints a QA report per paper; read it. Known limits:
  * a hyphen at a line end is dropped when the word without it occurs elsewhere in the paper, else kept only for a few
    compound starters (so-, long-, well-...); the unconfirmed joins are listed in the report for a quick look;
  * letters with a dot above (Armenian transliteration, ṙ) are rebuilt by attaching the dot to the r/R next to it;
    other dotted letters, if any, are listed in the report (`dotted_letters`) for a look;
  * italic footnote lines can lose word spaces ("HistoiredeMār...", see `words_over_30_chars`);
  * a note whose text is missing in the PDF stays (almost) empty (see `notes_shorter_than_5_chars`);
  * paragraph breaks come from indentation and sentence ends; block quotes are not recognised as such.
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

CONFIG = {
    "alford2026": dict(
        pdf="ALFORD - Borderland Lords (JLAIBS 2026).pdf",
        body=10.5,
        note=8.5,
        marker_below=7.5,  # font sizes: text, footnotes, footnote markers are smaller than this
        head_top=80,
        foot_top=640,  # running header above / page number below these y positions
        margin_x=450,  # right margin holds line numbers 1,5,10,...
        page1_text_size=10.5,
        page1_drop_sizes=(9.0,),
        page1_drop_below=None,
    ),  # 9.0 = journal footer block
    "nechaeva2017": dict(
        pdf="SLA0104_03_Nechaeva_SevenHellenesandOneChristian.pdf",
        body=10.7,
        note=8.2,
        marker_below=7.5,
        head_top=70,
        foot_top=625,
        margin_x=None,
        page1_text_size=8.2,
        page1_drop_sizes=(),
        page1_drop_below=540,
    ),  # abstract is set in 8.2, copyright block below 540
}

HYPHENS = "-­‐‑"
COMPOUND_FIRST = {
    "so",
    "long",
    "short",
    "well",
    "ill",
    "self",
    "half",
    "high",
    "low",
    "old",
}  # so-called, long-term ...
SENTENCE_END = tuple('.?!:;)]"”’»')


# ---------------------------------------------------------------- pure text functions
def fix_chars(s):
    """Private-use digits (U+F730..F739 = '0'..'9', and punctuation) back to ASCII, ligatures to letters,
    spacing dot-above glued to its neighbours (the PDF writes R˙ewan as 'R ˙ ewan')."""
    s = "".join(chr(ord(c) - 0xF700) if 0xF720 <= ord(c) <= 0xF77E else c for c in s)

    def dot(
        m,
    ):  # the PDF puts the dot before or after its letter; in Armenian transliteration it is r/R
        a, b = m.group(1), m.group(2)
        return a + b + "̇" if b in "rR" and a not in "rR" else a + "̇" + b

    s = re.sub(
        r"(\w)\s*˙\s*(\w)", dot, s
    )  # must run before NFKC, which would turn U+02D9 into ' ' + U+0307
    return unicodedata.normalize("NFKC", s)


def tidy(text):
    """The PDF sets a dot-above as a separate glyph after a gap ('Ar ̇awełean', 'R ̇ ewan'): glue it to the letter before
    it, close the gap behind it, and compose (r + U+0307 -> ṙ)."""
    dot = "̇"
    text = re.sub(r"(\w)\s+" + dot, lambda m: m.group(1) + dot, text)
    text = re.sub(dot + r"\s+(?=[^\W\d_])", dot, text)
    return unicodedata.normalize("NFC", text)


def is_heading_text(text, bold):
    letters = [c for c in text if c.isalpha()]
    caps = len(letters) >= 4 and all(c.isupper() for c in letters)
    return len(text) <= 90 and not text.endswith(".") and (caps or bold)


def _word_key(w):
    return re.sub(r"[^\w]", "", w).casefold()


def join_lines(lines, vocab, log):
    """Join physical lines into one string. A hyphen at a line end is dropped when the word without it is
    attested elsewhere in the paper (or the hyphenated form is not); an attested hyphenated compound keeps it."""
    out = lines[0]
    for nxt in lines[1:]:
        if out and out[-1] in HYPHENS and nxt and nxt[0].islower():
            left = re.search(r"(\w+)$", out[:-1])
            right = re.match(r"(\w+)", nxt)
            if left and right:
                merged, hy = (
                    _word_key(left.group(1) + right.group(1)),
                    _word_key(left.group(1)) + "-" + _word_key(right.group(1)),
                )
                if left.group(1).lower() in COMPOUND_FIRST and not vocab.get(merged):
                    log.append(("kept", left.group(1) + "-" + right.group(1), False))
                    out = out + nxt
                    continue
                if vocab.get(merged) or not vocab.get(hy):
                    log.append(
                        (
                            "joined",
                            left.group(1) + right.group(1),
                            bool(vocab.get(merged)),
                        )
                    )
                    out = out[:-1] + nxt
                    continue
                log.append(("kept", left.group(1) + "-" + right.group(1), True))
                out = out + nxt
                continue
        if out and out[-1] in HYPHENS and nxt and nxt[0].isupper():
            out += nxt
            continue
        if out.endswith("–") and nxt[:1].isdigit():
            out += nxt
            continue
        out += " " + nxt
    return re.sub(r"[ \t]+", " ", out).strip()


# ---------------------------------------------------------------- layout -> lines (needs pdfplumber)
def page_lines(page, cfg):
    """One dict per physical text line of the page, margin numbers / headers / footers already removed and
    footnote markers (superscript digits) split off into line['markers']."""
    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=3,
        keep_blank_chars=False,
        extra_attrs=["size", "fontname"],
    )
    keep = []
    for w in words:
        if cfg["margin_x"] and w["x0"] >= cfg["margin_x"]:
            continue
        if not (cfg["head_top"] <= w["top"] <= cfg["foot_top"]):
            continue
        w = dict(w, text=fix_chars(w["text"]))
        if w["text"].strip():
            keep.append(w)
    marks = [w for w in keep if w["size"] < cfg["marker_below"]]
    main = sorted(
        (w for w in keep if w["size"] >= cfg["marker_below"]),
        key=lambda w: (w["top"], w["x0"]),
    )
    groups = []
    for w in main:
        if groups and abs(w["top"] - groups[-1][0]["top"]) <= 6:
            groups[-1].append(w)
        else:
            groups.append([w])
    near = [[] for _ in groups]  # footnote markers belonging to each line
    for m in marks:
        if not re.sub(r"\D", "", m["text"]) or not groups:
            continue
        i = min(range(len(groups)), key=lambda k: abs(groups[k][0]["top"] - m["top"]))
        if abs(groups[i][0]["top"] - m["top"]) < 10:
            near[i].append(m)
    lines = []
    for g, ms in zip(groups, near):
        g.sort(key=lambda w: w["x0"])
        text, prev = "", None
        for w in g:
            if prev is not None and w["x0"] - prev["x1"] > 1.0:
                marker_between = any(
                    prev["x1"] - 1 <= m["x0"] <= w["x0"] + 0.5 for m in ms
                )
                if not (marker_between and w["text"][0] in ".,;:)]!?\u201d\u2019"):
                    text += " "
            text += w["text"]
            prev = w
        sizes = collections.Counter()
        for w in g:
            sizes[round(w["size"], 1)] += len(w["text"])
        size = sizes.most_common(1)[0][0]
        ms.sort(key=lambda m: m["x0"])
        lines.append(
            dict(
                text=text.strip(),
                x0=g[0]["x0"],
                x1=max(w["x1"] for w in g),
                top=g[0]["top"],
                size=size,
                bold=all(
                    "Bold" in w["fontname"] or w["fontname"].endswith(".B") for w in g
                ),
                dropcap=g[0]["size"] > size + 3,
                markers=[int(re.sub(r"\D", "", m["text"])) for m in ms],
                marker_x=[m["x0"] for m in ms],
            )
        )
    return lines


# ---------------------------------------------------------------- lines -> paragraphs (pure)
def classify(line, page_no, cfg, state):
    s, t = line["size"], line["text"]
    if page_no == 1 and not state["abstract"]:
        if re.match(r"(?i)abstract\b", t):
            state["abstract"] = True
        elif s >= 12:
            return "title"
        else:
            return "front"  # author, affiliation: not text of the paper
    if page_no == 1:
        if any(abs(s - d) < 0.1 for d in cfg["page1_drop_sizes"]):
            return "drop"
        if cfg["page1_drop_below"] and line["top"] >= cfg["page1_drop_below"]:
            return "drop"
        if abs(s - cfg["page1_text_size"]) <= 0.3:
            return "body"
    if abs(s - cfg["note"]) <= 0.15:
        return "note"
    if abs(s - cfg["body"]) <= 0.3:
        return "heading" if is_heading_text(t, line["bold"]) else "body"
    return "heading"


def assemble(pages, cfg):
    """pages: [[line dict, ...], ...] in reading order -> dict(title, paragraphs, footnotes, headings, log)."""
    state = {"abstract": False}
    title_lines, body, notes = [], [], {}
    last_note, headings = None, []
    for pno, lines in enumerate(pages, 1):
        tagged = [(classify(l, pno, cfg, state), l) for l in lines]
        bl = [l for k, l in tagged if k == "body"]
        left = (
            collections.Counter(round(l["x0"]) for l in bl).most_common(1)[0][0]
            if bl
            else 0
        )
        right = sorted(l["x1"] for l in bl)[int(0.9 * (len(bl) - 1))] if bl else 0
        prev_kind = None
        for kind, l in tagged:
            l = dict(l, page=pno, left=left, right=right)
            if kind == "title":
                title_lines.append(l)
            elif kind == "heading":
                if (
                    prev_kind == "heading"
                    and body
                    and body[-1]["kind"] == "heading"
                    and l["top"] - body[-1]["top"] < 20
                ):
                    body[-1]["text"] += " " + l["text"]
                else:
                    body.append(dict(l, kind="heading"))
            elif kind == "body":
                body.append(dict(l, kind="body"))
            elif kind == "note":
                m = re.match(r"^(\d{1,3})\.?\s+(?=\S)", l["text"])
                expect = (last_note or 0) + 1
                # some journals set the note number as a small superscript left of the text (Alford): it is a marker
                lead = [d for d, x in zip(l["markers"], l["marker_x"]) if x < l["x0"]]
                if lead and lead[0] == expect:
                    last_note = expect
                    notes[last_note] = [l["text"]]
                elif m and int(m.group(1)) == expect:
                    last_note = expect
                    notes[last_note] = [l["text"][m.end() :]]
                elif l["text"].startswith("*"):
                    last_note = None  # author's starred note: not part of the paper
                elif last_note is not None:
                    notes[last_note].append(l["text"])
            prev_kind = kind

    paragraphs, cur, section, reset_next = [], None, None, False

    def close():
        nonlocal cur
        if cur:
            paragraphs.append(cur)
            cur = None

    if state["abstract"]:
        section = "abstract"
    for l in body:
        if l["kind"] == "heading":
            close()
            section = l["text"]
            headings.append(l["text"])
            continue
        starts = cur is None
        if cur is not None:
            prev = cur["lines"][-1]
            indent = l["x0"] - l["left"] > 6
            short = prev["x1"] < prev["right"] - 25
            ends = prev["text"].rstrip().endswith(SENTENCE_END)
            bullet = re.match(r"^[-\u2022\u2013]\s", l["text"])
            was_bullet = re.match(r"^[-\u2022\u2013]\s", cur["lines"][0]["text"])
            if (
                l["dropcap"]
                or re.match(r"(?i)keywords\b", l["text"])
                or bullet
                or was_bullet
                and short
                and l["text"][:1].isupper()
                or ends
                and (indent or (short and l["text"][:1].isupper()))
            ):
                starts = True
        if starts:
            close()
            if l["dropcap"]:
                section = None
            if reset_next:
                section, reset_next = None, False
            if re.match(r"(?i)keywords\b", l["text"]):
                reset_next = True
            cur = dict(
                section=section
                if not re.match(r"(?i)keywords\b", l["text"])
                else "keywords",
                page=l["page"],
                lines=[],
                notes=[],
            )
        cur["lines"].append(l)
        cur["notes"] += l["markers"]
    close()
    return dict(
        title_lines=title_lines, paragraphs=paragraphs, notes=notes, headings=headings
    )


def finalize(parsed, doc_id, source):
    """Join lines, de-hyphenate with the paper's own vocabulary, split the glued 'KEYWORDS' tail, number paragraphs."""
    interior = collections.Counter()
    for p in parsed["paragraphs"]:
        for l in p["lines"]:
            for w in re.findall(r"[\w-]+", l["text"])[1:-1]:
                interior[_word_key(w)] += 1
    log, paras = [], []
    for p in parsed["paragraphs"]:
        text = tidy(join_lines([l["text"] for l in p["lines"]], interior, log))
        pieces = [(p["section"], text)]
        m = re.search(r"\s(KEYWORDS|Keywords\.?)\s", text)
        if m and p["page"] == 1 and p["section"] == "abstract":
            pieces = [
                ("abstract", text[: m.start()].strip()),
                ("keywords", text[m.start() :].strip()),
            ]
        for sec, t in pieces:
            if t:
                paras.append(
                    dict(
                        section=sec,
                        page=p["page"],
                        text=t,
                        notes=sorted(set(p["notes"])),
                    )
                )
    for i, p in enumerate(paras, 1):
        p["id"] = f"p{i:03d}"
    foot = {
        str(n): tidy(join_lines(parts, interior, log))
        for n, parts in parsed["notes"].items()
    }
    tl = parsed["title_lines"]
    title = ""
    for i, l in enumerate(tl):
        if i and l["top"] - tl[i - 1]["top"] > 25:
            break
        title += (" " if title else "") + l["text"]
    return dict(
        doc_id=doc_id,
        title=title,
        source=source,
        paragraphs=[
            dict(
                id=p["id"],
                text=p["text"],
                section=p["section"],
                page=p["page"],
                notes=p["notes"],
            )
            for p in paras
        ],
        footnotes=foot,
        headings=parsed["headings"],
    ), log


def qa(doc, log):
    ps, foot = doc["paragraphs"], doc["footnotes"]
    refs = {n for p in ps for n in p["notes"]}
    defined = {int(k) for k in foot}
    text = " ".join(p["text"] for p in ps) + " " + " ".join(foot.values())
    words = re.findall(r"\S+", text)
    odd = collections.Counter(
        c for c in text if (0xE000 <= ord(c) <= 0xF8FF) or c in "�\u0000"
    )
    return {
        "paragraphs": len(ps),
        "words_in_paragraphs": sum(len(p["text"].split()) for p in ps),
        "footnotes": len(foot),
        "highest_marker": max(refs) if refs else 0,
        "markers_without_footnote": sorted(refs - defined)[:20],
        "footnotes_never_cited": sorted(defined - refs)[:20],
        "notes_shorter_than_5_chars (empty in the PDF itself?)": [
            k for k, v in foot.items() if len(v) < 5
        ],
        "headings": len(doc["headings"]),
        "short_paragraphs(<40 chars)": [
            p["text"][:50] for p in ps if len(p["text"]) < 40
        ][:8],
        "longest_paragraph_chars": max((len(p["text"]) for p in ps), default=0),
        "words_over_30_chars (spaces lost?)": sorted({w for w in words if len(w) > 30})[
            :8
        ],
        "private_use_or_replacement_chars_left": dict(odd),
        "year_like_numbers": len(re.findall(r"\b[3-7]\d{2}\b|\b1[5-9]\d{2}\b", text)),
        "dotted_letters": sorted(
            {
                w.strip("“”‘’()[],.;:")
                for w in words
                if any(
                    unicodedata.combining(c) and c == "̇"
                    for c in unicodedata.normalize("NFD", w)
                )
            }
        )[:12],
        "hyphen_joins": sum(1 for k, *_ in log if k == "joined"),
        "hyphens_kept": [w for k, w, _ in log if k == "kept"][:8],
        "joins_not_confirmed_by_vocabulary": [
            w for k, w, ok in log if k == "joined" and not ok
        ][:15],
    }


def run(paper_dir, out_dir, only=None):
    import pdfplumber

    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, cfg in CONFIG.items():
        if only and doc_id != only:
            continue
        pdf_path = paper_dir / cfg["pdf"]
        if not pdf_path.exists():
            print(f"{doc_id}: {pdf_path} not found, skipped")
            continue
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page_lines(p, cfg) for p in pdf.pages]
        doc, log = finalize(assemble(pages, cfg), doc_id, cfg["pdf"])
        (out_dir / f"{doc_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n=== {doc_id}: {doc['title']}")
        for k, v in qa(doc, log).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--papers", default="papers")
    ap.add_argument("--out", default="texts")
    ap.add_argument("--only")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(Path(a.papers), Path(a.out), a.only)
