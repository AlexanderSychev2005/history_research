"""Stage 1, mentions. Code proposes spans (runs of capitalised words), Jev judges them; Jev returns no spans itself.

For a run in a sentence:
  is it a name?            Noul, per span
  what kind?               Choice (person, place, people or state, group, written work, event)
  "A of B"                 Choice: one name (Bank of England) or a person and a separate place (Hugh of Lincoln)
  title in front?          Choice over the suffixes of a person's run: "King Henry II" -> "Henry II"
  lowercase head noun?     Noul: "Tudor dynasty" as one name
Code alone drops function words, month names and abbreviations inside brackets ("(Mart. Grig. 8)").
All questions of a sentence share one state and go out in one batch; the later ones are asked speculatively and
`read` decides which answers count.
"""
import re

from questions import KINDS, T_NAME, both_orders, choice_probs, is_name, kind, name_boundary, of_reading, with_head_noun

WORD = re.compile(r"[^\W\d_](?:[\w-]|[‘’ʻ](?=\w))*‘?")
NOT_NAMES = set("""A An The This That These Those He She It They We I You His Her Its Their Our My In On At By For From
With Of To As After Before During Around According However While Although Though Such Most Still Another Other Others
One Two Three Four Five Six Seven Some Many Both Each Every All Any Nothing There Here When Where What Who Which Whose
And But Or Nor If Then Thus Therefore Moreover Also Yet Since Because Not No Only Even AD BC CE BCE January February
March April May June July August September October November December""".split())
HEAD_NOUNS = set("""family dynasty clan house rebellion revolt uprising war wars campaign treaty peace council synod
empire kingdom church""".split())
ROMAN = re.compile(r"^[IVX]+$")


def sentences(text):
    """(start, end) of each sentence; 'c. 470' and 'St. John' do not end one, nor does a full stop inside brackets."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?”])\s+(?=\S)", text):
        so_far, nxt = text[start:m.start()], text[m.end()]
        if ((nxt.isupper() or nxt in "“‘(")                      # isupper: a sentence may start with Ł or Š
                and not re.search(r"\b(?:c|St|cf|ed|eds|trans|vol|no|pp?|r)\.$", so_far)
                and so_far.count("(") <= so_far.count(")")):
            out.append((start, m.start()))
            start = m.end()
    out.append((start, len(text)))
    return out


def runs(sent):
    """Capitalised runs of a sentence: [{'segs': [(start, end)], 'noun': (start, end) or None}]. A run has several
    segments when 'of' / 'of the' joins them; 'noun' is a lowercase head noun right after the run."""
    words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(sent)]
    groups, cur = [], []                                         # (words of one segment, index of its last word)
    for i, (s, e, w) in enumerate(words + [(None, None, None)]):
        cap = w is not None and w[0].isupper() and w not in NOT_NAMES
        if cap and sent[e:e + 1] == "." and sent[:s].count("(") > sent[:s].count(")") and sent[e + 1:].strip(" )"):
            cap = False                                          # "(Mart. Grig. 8)": an abbreviation in a citation
        if cur and (not cap or not sent[cur[-1][1]:s].isspace()):
            groups.append((cur, i - 1))
            cur = []
        if cap:
            cur.append((s, e, w))
    out = []
    for toks, last in groups:
        s, e = toks[0][0], toks[-1][1]
        e -= 2 if sent[s:e].endswith("’s") else 0
        nxt = words[last + 1] if last + 1 < len(words) else None
        noun = (nxt[0], nxt[1]) if nxt and nxt[2] in HEAD_NOUNS and sent[e:nxt[0]].isspace() else None
        if out and out[-1]["noun"] is None and re.fullmatch(r"\s+of\s+(?:the\s+)?", sent[out[-1]["segs"][-1][1]:s]):
            out[-1]["segs"].append((s, e))
            out[-1]["noun"] = noun
        else:
            out.append({"segs": [(s, e)], "noun": noun})
    return out


def suffixes(sent, s, e):
    """Spans from each word of sent[s:e] to its end, longest first; a bare numeral (II) is not offered."""
    starts = [s + m.start() for m in WORD.finditer(sent[s:e])]
    return [(x, e) for x in starts if not ROMAN.match(sent[x:e])]


def plan(sent):
    """-> (the runs of a sentence, every question about them)"""
    found, qs = runs(sent), {}
    for r in found:
        spans = set()
        for s, e in r["segs"]:
            parts = suffixes(sent, s, e)
            spans.update(parts)
            if len(parts) > 1:
                qs.update(both_orders(f"cut|{s}|{e}", name_boundary([sent[x:e] for x, _ in parts])))
        whole = (r["segs"][0][0], r["segs"][-1][1])
        if len(r["segs"]) > 1:
            spans.add(whole)
            a, b = sent[r["segs"][0][0]:r["segs"][0][1]], sent[r["segs"][1][0]:whole[1]]
            qs.update(both_orders(f"of|{whole[0]}|{whole[1]}", of_reading(a, b)))
        if r["noun"]:
            s, e = r["segs"][-1][0], r["noun"][1]
            spans.add((s, e))
            qs[f"noun|{s}|{e}"] = with_head_noun(sent[s:r["segs"][-1][1]], sent[r["noun"][0]:e])
        for s, e in spans:
            qs[f"name|{s}|{e}"] = is_name(sent[s:e])
            qs.update(both_orders(f"kind|{s}|{e}", kind(sent[s:e])))
    return found, qs


def read(sent, found, ans, t_name=T_NAME):
    """The truth table: which spans of each run become mentions."""
    out = []

    def mention(s, e, p=None):
        kinds = choice_probs(ans, f"kind|{s}|{e}")
        k = max(kinds, key=kinds.get)
        return {"start": s, "end": e, "text": sent[s:e], "type": KINDS[k][1], "kind": k,
                "p": round(ans[f"name|{s}|{e}"]["noul"] if p is None else p, 4),
                "kinds": {x: round(v, 4) for x, v in kinds.items()}}

    def segment(s, e):
        """a run without 'of': the whole, or for a person the name without the title in front"""
        m = mention(s, e)
        if m["kind"] == "person" and f"cut|{s}|{e}" in ans:
            cut = choice_probs(ans, f"cut|{s}|{e}")
            m = mention(e - len(max(cut, key=cut.get)), e)
        return m

    for r in found:
        segs = r["segs"]
        whole = (segs[0][0], segs[-1][1])
        if len(segs) > 1 and choice_probs(ans, f"of|{whole[0]}|{whole[1]}")["one_name"] >= 0.5:
            names = [mention(*whole)]
        else:
            names = [segment(s, e) for s, e in segs]
        if r["noun"]:
            s, e = segs[-1][0], r["noun"][1]
            p = ans[f"noun|{s}|{e}"]["noul"]
            if p >= 0.5 and names[-1]["start"] >= s:             # the last segment and its noun are one name
                names[-1] = mention(s, e, p=min(p, ans[f"name|{s}|{e}"]["noul"]))
        out += [m for m in names if m["p"] >= t_name]
    return out


def with_sentence(text, start):
    return next((a, b) for a, b in sentences(text) if a <= start < b)


def gold_mentions(jev, paragraphs, gold):
    """The gold spans with the kinds from Jev: the input for testing the later stages alone."""
    text = dict(paragraphs)
    out = [{"para": m["para"], "start": m["start"], "end": m["end"], "text": m["text"], "p": 1.0,
            "sentence": list(with_sentence(text[m["para"]], m["start"]))} for m in gold["mentions"]]
    jobs = [({"sentence": text[m["para"]][m["sentence"][0]:m["sentence"][1]]}, both_orders("kind", kind(m["text"])))
            for m in out]
    for m, ans in zip(out, jev.ask_many(jobs)):
        kinds = choice_probs(ans, "kind")
        k = max(kinds, key=kinds.get)
        m.update(kind=k, type=KINDS[k][1], kinds={x: round(v, 4) for x, v in kinds.items()})
    return out


def find_mentions(jev, paragraphs, t_name=T_NAME):
    """paragraphs: [(para_id, text)] -> [{para, start, end, text, type, kind, kinds, p, sentence}]"""
    jobs, meta = [], []
    for pid, text in paragraphs:
        for s0, s1 in sentences(text):
            found, qs = plan(text[s0:s1])
            if qs:
                jobs.append(({"sentence": text[s0:s1]}, qs))
                meta.append((pid, s0, s1, found))
    out = []
    for (pid, s0, s1, found), (state, _), ans in zip(meta, jobs, jev.ask_many(jobs)):
        for m in read(state["sentence"], found, ans, t_name):
            out.append({"para": pid, **m, "start": s0 + m["start"], "end": s0 + m["end"], "sentence": [s0, s1]})
    return sorted(out, key=lambda m: (m["para"], m["start"]))
