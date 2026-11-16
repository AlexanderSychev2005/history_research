"""Stage 2, entities: group the written forms of one entity, then link each entity to YAGO 4.5.

Jev cannot write a name down, so it cannot guess "Ghazar Parpetsi" for "Łazar P‘arpec‘i" the way an LLM does. Code
finds the candidates and Jev judges them:
  forms       code proposes pairs (Łazar / Łazar P‘arpec‘i; Gorgān (Latin. Hyrcania)), a Noul says whether they are one
  search      full-text search of English Wikipedia for the written forms, for "Isidore of Gaza" with the place,
              and for scholarly transliterations also in the popular spelling (Aršawir -> Arshavir, by rule);
              a YAGO 4.5 id is the Wikipedia title with "_" for punctuation
  YAGO        what YAGO holds on each candidate (sources.py): its classes (none: it is not in YAGO, as an ethnic
              group is not), its years, its alternate names. The names go into the description that Jev reads
              ("Khosrow I ... also called Chosroes I")
  gates       Jev says of what kind the entity is and in which century it lived, from every sentence that names it,
              and the same about each candidate from its description. Code compares and drops a candidate of
              another kind (a battle for a town) or, for persons and events, of another period (Yazdegerd III for
              a Yazdgerd of the 5th century). YAGO's years and its class Person replace Jev's reading where YAGO
              has them; its other classes do not (a treaty is a CreativeWork there and an event to a reader).
              Jev compares dates badly, so it only reads them and code compares
  identity    link_any (Noul, the companion) x link_which (Choice, both orders) x link_is (Noul per candidate),
              geometric mean as in the relation stage; below the threshold the entity stays unlinked
  link graph  entities of one paper are linked to each other in Wikipedia too. The articles that the sure links
              point to, about 5,000 titles per paper, become candidates for every entity that is not sure: Jev picks
              among the bare titles, 120 at a time, the winners meet again (a tournament), and the five that are
              left go through YAGO, gates and identity like any other candidate
ablation.py switches the parts on one at a time.
"""
import re
import unicodedata

from questions import (CENTURIES, KINDS, T_LINK, T_SAME, both_orders, century_of, choice_probs, dated, entry_century,
                       entry_dated, entry_kind, link_any, link_is, link_which, other_name, pick_article, same_entity)
from sources import PER_QUERY, yago_dump_ids, yago_id

PARTS = frozenset({"dossier", "kind", "period", "graph", "yago", "aliases"})
MAX_CANDIDATES = 12     # from search, per entity
SURE = 0.9              # a link this probable is an anchor: the articles it points to become candidates
KIND_GATE, PERIOD_GATE = 0.2, 0.15      # a candidate is dropped if kind or period agree with less probability
NOT_ARTICLES = re.compile(r"^(?:Lists? of|Index of|Outline of|Timeline of)\b|\(disambiguation\)|^\d|:")

_FOLD = str.maketrans({"ł": "l", "Ł": "L", "‘": "", "’": "", "ʻ": "", "ǰ": "j", "ə": "e"})
_POPULAR = [("c‘", "ts"), ("č‘", "ch"), ("‘", ""), ("ʻ", ""), ("ł", "gh"), ("Ł", "Gh"), ("š", "sh"), ("Š", "Sh"),
            ("č", "ch"), ("Č", "Ch"), ("ž", "zh"), ("Ž", "Zh"), ("ǰ", "j"), ("x", "kh"), ("w", "v"), ("c", "ts")]


def fold(s):
    """without diacritics and transliteration marks: Łazar P‘arpec‘i -> Lazar Parpeci"""
    return "".join(c for c in unicodedata.normalize("NFKD", s.translate(_FOLD)) if not unicodedata.combining(c))


def popular(s):
    """A name in the scholarly romanisation of Armenian or Persian (Hübschmann-Meillet: ł, š, c‘, w) in the spelling
    that encyclopedias use: Łazar P‘arpec‘i -> Ghazar Parpetsi, Aršawir -> Arshavir. Other names are returned as is."""
    if not any(ch in "łŁšŠčČžŽǰēə‘ʻ" for ch in s):
        return s
    for a, b in _POPULAR:
        s = s.replace(a, b)
    return fold(s)


def tokens(text):
    return set(re.findall(r"\w+", fold(text).lower()))


# ------------------------------------------------------------------ forms of one entity

def cluster(jev, paragraphs, mentions):
    """Sets m['entity'] for every mention and returns [{eid, label, forms, kind, type, kinds}]."""
    text = dict(paragraphs)
    key = lambda m: fold(m["text"]).lower()
    first = {}                                                   # form -> its first mention
    for m in mentions:
        first.setdefault(key(m), m)
    sentence = lambda m: text[m["para"]][m["sentence"][0]:m["sentence"][1]]

    jobs, pairs = [], []                                         # pairs: (form, the form it may belong to)
    # "Gorgān (Latin. Hyrcania", "Domitziolus (or Domnentiolus)": a second name in brackets
    for a, b in zip(mentions, mentions[1:]):
        if (a["para"], a["sentence"]) == (b["para"], b["sentence"]) and re.fullmatch(
                r"\s*\((?:[A-Za-z]+\.?\s+){0,2}", text[a["para"]][a["end"]:b["start"]]):
            jobs.append(({"sentence": sentence(a)}, {"same": other_name(a["text"], b["text"])}))
            pairs.append((key(b), key(a)))
    # "Łazar" and "Łazar P‘arpec‘i": a shorter form of a longer name, judged in every paragraph that has the short one
    for s, short in first.items():
        for l, long_ in first.items():
            if tokens(short["text"]) < tokens(long_["text"]):
                for pid in sorted({m["para"] for m in mentions if key(m) == s}):
                    state = {"paragraph": text[pid]}
                    if long_["text"] not in text[pid]:
                        state["earlier_sentence"] = sentence(long_)
                    jobs.append((state, {"same": same_entity(short["text"], long_["text"])}))
                    pairs.append((s, l))
    votes = {}
    for pair, ans in zip(pairs, jev.ask_many(jobs)):
        votes.setdefault(pair, []).append(ans["same"]["noul"])

    parent = {k: k for k in first}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    best = {}                                                    # a form joins the form it most probably belongs to
    for (s, l), ps in votes.items():
        p = sum(ps) / len(ps)
        if p >= T_SAME and p > best.get(s, (0, None))[0]:
            best[s] = (p, l)
    for s, (_, l) in best.items():
        if root(l) != root(s):
            parent[root(s)] = root(l)

    groups = {}
    for m in mentions:
        groups.setdefault(root(key(m)), []).append(m)
    entities = []
    for i, ms in enumerate(sorted(groups.values(), key=lambda ms: (ms[0]["para"], ms[0]["start"]))):
        forms = sorted({m["text"] for m in ms}, key=lambda f: (-len(f), f))
        kinds = {k: round(sum(m["kinds"][k] for m in ms) / len(ms), 4) for k in KINDS}
        k = max(kinds, key=kinds.get)
        entities.append({"eid": f"e{i}", "label": forms[0], "forms": forms, "kind": k, "type": KINDS[k][1],
                         "kinds": kinds})
        for m in ms:
            m["entity"] = f"e{i}"
    return entities


# ------------------------------------------------------------------ candidates

def queries(entity, mentions, text):
    """Search strings for an entity: "A of B" where the text has it, the longest form, any form that is not just a
    part of it ("Hyrcania" beside "Gorgān"), and each of these in the popular spelling. The search engine ignores
    plain diacritics itself."""
    out, used = [], []
    for f in entity["forms"]:
        if any(tokens(f) <= tokens(u) for u in used) or len(used) == 3:
            continue
        used.append(f)
        for m in mentions:
            if m["text"] == f:
                of = re.match(r"\s+of\s+(?:the\s+)?([^\s,.;:()—–]+)", text[m["para"]][m["end"]:])
                if of and of.group(1)[0].isupper():
                    out.append(f"{f} of {of.group(1).rstrip('’')}")
        out.append(f)
    return list(dict.fromkeys(out + [popular(q) for q in out]))


def shortlist(wiki, qs):
    """The results of all queries, taken in turn (the best of each query first), without repeats."""
    lists, out = [wiki.search(q) for q in qs], {}
    for rank in range(PER_QUERY):
        for hits in lists:
            if rank < len(hits):
                out.setdefault(hits[rank]["title"], dict(hits[rank]))
    return list(out.values())[:MAX_CANDIDATES]


def tournament(jev, targets, pool, size=120, keep=3, final=5):
    """targets: [(context, name)]. Jev picks among article titles alone, `size` at a time; the winners of each group
    meet again until one group is left. -> for each target the `final` best titles."""
    current = [list(pool) for _ in targets]
    while True:
        last = all(len(c) <= size for c in current)
        jobs, meta = [], []
        for i, (ctx, name) in enumerate(targets):
            for s in range(0, len(current[i]), size):
                chunk = current[i][s:s + size]
                jobs.append(({"context": ctx}, both_orders("pick", pick_article(name, chunk))))
                meta.append((i, chunk))
        winners = [[] for _ in targets]
        for (i, chunk), ans in zip(meta, jev.ask_many(jobs)):
            p = choice_probs(ans, "pick")
            winners[i] += sorted(chunk, key=lambda t: -p[t])[:final if last else keep]
        if last:
            return winners
        current = winners


# ------------------------------------------------------------------ gates

COARSE = {"person": "person", "place": "place or group", "people_or_state": "place or group",
          "group": "place or group", "written_work": "work", "event": "event"}


def one_time(kinds):
    """a person or an event: something with a date, unlike a place or a state, which lasts"""
    return COARSE[max(kinds, key=kinds.get)] in ("person", "event")


def same_kind(e, reading):
    """probability that entity and candidate are the same sort of thing (person, place or group, work, event)"""
    return sum(p * q for k, p in e["kinds"].items() for j, q in reading["kinds"].items() if COARSE[k] == COARSE[j])


def same_period(e, reading):
    """probability that the two centuries, as Jev read them, are equal or adjacent"""
    return sum(e["century"][c] * reading["century"][d] for i, c in enumerate(CENTURIES)
               for j, d in enumerate(CENTURIES) if abs(i - j) <= 1)


def lived_then(e, years):
    """probability that the entity's century is within a century of the years YAGO gives for a candidate"""
    first = lambda i: (i - 6) * 100                              # CENTURIES[6] is the 1st century: the years 0 to 99
    return sum(e["century"][c] for i, c in enumerate(CENTURIES)
               if first(i) <= years[1] + 100 and first(i) + 99 >= years[0] - 100)


def gate(e, c, reading, parts):
    """Why candidate c cannot be entity e, or None. `reading` is Jev's reading of the candidate's description."""
    dated = one_time(reading["kinds"])
    if "yago" in parts:
        if not c["classes"]:
            return "not in YAGO"
        is_person = "schema:Person" in c["classes"]
        if (is_person and e["kinds"]["person"] < 0.2) or (not is_person and e["kinds"]["person"] > 0.8):
            return "another kind"
        dated = dated or is_person
    if "kind" in parts and same_kind(e, reading) < KIND_GATE:
        return "another kind"
    if "period" in parts and one_time(e["kinds"]) and dated and e["dated"] >= 0.5:
        if "yago" in parts and c["years"]:
            if lived_then(e, c["years"]) < PERIOD_GATE:
                return "another period"
        elif reading["dated"] >= 0.5 and same_period(e, reading) < PERIOD_GATE:
            return "another period"
    return None


# ------------------------------------------------------------------ links

def name_of(e):
    return e["label"] + (f"” (also written “{e['forms'][1]}" if len(e["forms"]) > 1 else "")


def link(jev, wiki, yago, paragraphs, mentions, entities, threshold=T_LINK, parts=PARTS):
    """Adds to every entity: yago, wikipedia, wikidata, p_link, century and its candidates."""
    text, dump, readings = dict(paragraphs), yago_dump_ids(), {}
    of = {e["eid"]: [m for m in mentions if m["entity"] == e["eid"]] for e in entities}

    # what Jev reads about an entity: every sentence of the document that names it (at most six), or without the
    # dossier the one paragraph where its longest form first stands
    ctx = {}
    for e in entities:
        if "dossier" in parts:
            ctx[e["eid"]] = list(dict.fromkeys(text[m["para"]][m["sentence"][0]:m["sentence"][1]]
                                               for m in of[e["eid"]]))[:6]
        else:
            ctx[e["eid"]] = text[next(m for m in of[e["eid"]] if m["text"] == e["label"])["para"]]

    jobs = [({"context": ctx[e["eid"]]}, {**both_orders("century", century_of(name_of(e))), "dated": dated(name_of(e))})
            for e in entities]
    for e, ans in zip(entities, jev.ask_many(jobs)):
        e["century"], e["dated"] = choice_probs(ans, "century"), ans["dated"]["noul"]
        e["candidates"] = shortlist(wiki, queries(e, of[e["eid"]], text))

    def judge(es):
        """YAGO, gates and identity for the candidates of some entities"""
        new = [c for e in es for c in e["candidates"] if "out" not in c]
        records = yago.entities([c["title"] for c in new]) if {"yago", "aliases"} & parts else {}
        for c in new:
            r = records.get(c["title"]) or {"qid": None, "classes": [], "years": None, "aliases": []}
            c.update(yago=yago_id(c["title"]), wikidata=r["qid"], classes=r["classes"], years=r["years"])
            if "aliases" in parts and r["aliases"]:
                c["description"] = f"{c['description']}; also called {', '.join(r['aliases'][:6])}"
        unread = {c["title"]: c for c in new if c["title"] not in readings}
        jobs = [({"entry": {"title": t, "description": c["description"]}},
                 {**both_orders("kind", entry_kind()), **both_orders("century", entry_century()), "dated": entry_dated()})
                for t, c in unread.items()]
        for t, ans in zip(unread, jev.ask_many(jobs)):
            readings[t] = {"kinds": choice_probs(ans, "kind"), "century": choice_probs(ans, "century"),
                           "dated": ans["dated"]["noul"]}

        jobs, asked = [], []
        for e in es:
            e.update(yago=None, wikipedia=None, wikidata=None, p_link=0.0)
            for c in e["candidates"]:
                c["p"] = 0.0
                c["out"] = gate(e, c, readings[c["title"]], parts)
                if dump is not None and c["yago"] not in dump:
                    c["out"] = "not in YAGO"
            live = [c for c in e["candidates"] if not c["out"]]
            if live:
                name = name_of(e)
                qs = {"any": link_any(name),
                      **both_orders("which", link_which(name, [(c["title"], c["description"]) for c in live]))}
                for i, c in enumerate(live):
                    qs[f"is|{i}"] = link_is(name, c["title"], c["description"])
                jobs.append(({"context": ctx[e["eid"]], "candidates": {c["title"]: c["description"] for c in live}}, qs))
                asked.append((e, live))
        for (e, live), ans in zip(asked, jev.ask_many(jobs)):
            which = choice_probs(ans, "which")
            for i, c in enumerate(live):
                c["p"] = round((ans["any"]["noul"] * which[c["title"]] * ans[f"is|{i}"]["noul"]) ** (1 / 3), 4)
            top = max(live, key=lambda c: c["p"])
            e["p_link"] = top["p"]
            if top["p"] >= threshold:
                e.update(yago=top["yago"], wikipedia=top["title"], wikidata=top["wikidata"])

    judge(entities)
    if "graph" in parts:
        sure = [max(e["candidates"], key=lambda c: c["p"])["title"] for e in entities if e["p_link"] >= SURE]
        pool = sorted({t for a in sure for t in [a] + wiki.links(a) if not NOT_ARTICLES.search(t)})
        unsure = [e for e in entities if e["p_link"] < SURE]
        if pool and unsure:
            picks = tournament(jev, [(ctx[e["eid"]], name_of(e)) for e in unsure], pool)
            wiki.pages([t for titles in picks for t in titles])  # one batch; the loop below reads the cache
            for e, titles in zip(unsure, picks):
                have = {c["title"] for c in e["candidates"]}
                for p in wiki.pages(titles):                     # two titles may redirect to one article
                    if p["title"] not in have:
                        have.add(p["title"])
                        e["candidates"].append(dict(p, via="links"))
            judge(unsure)
    for e in entities:
        e["century"] = max(e["century"], key=e["century"].get) if e.pop("dated") >= 0.5 else None
    return entities
