"""Where the candidates and the knowledge about them come from: Wikipedia's search and links, and YAGO.

YAGO is used without its dump. It is built from Wikidata (github.com/yago-naga/yago-4.6): an entity is kept if one
of its Wikidata classes lies under a class of the hand-made schema (yago-schema.ttl: "each class defined here will
automatically draw in its subclasses from Wikidata"), and its names and facts are Wikidata's. So the schema plus
Wikidata answer, for any Wikipedia article, what a lookup in the dump would: is the entity in YAGO, of which class,
when did it exist, what else is it called, which facts does YAGO have about it.

Both sources are cached in jev/cache/ and asked one request at a time. Wikimedia allows a client that does not say
who it is 10 requests a minute, and 200 with a contact (an e-mail or URL) in the User-Agent: put yours in the
environment variable WIKIPEDIA_CONTACT to go faster. Nothing is sent about you otherwise.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PER_QUERY = 6                                   # search results kept per query
DATES = ("P569", "P570", "P571", "P576", "P580", "P582", "P585", "P1317", "P2031", "P2032", "P577")


def yago_id(title):
    """YAGO 4.5 names an entity by its English Wikipedia title: Robert W. Thomson -> yago:Robert_W_Thomson"""
    return "yago:" + re.sub(r"_+", "_", re.sub(r"[^\w-]", "_", title.replace("–", "-"))).strip("_")


def yago_dump_ids():
    """The ids of the YAGO 4.5 entity dump if YAGO_JSONL points to it (the file that LELA downloads), else None."""
    path = os.environ.get("YAGO_JSONL")
    if not path or not os.path.exists(path):
        if path:
            print(f"YAGO_JSONL: {path} does not exist, going on without the dump", file=sys.stderr)
        return None
    ids = set()
    for line in open(path, encoding="utf-8"):
        i = line.find('"id": "')
        ids.add(line[i + 7:line.find('"', i + 7)])
    return ids


class Source:
    """A JSON cache on disk and paced GET / POST requests that wait when the server says so."""

    def __init__(self, cache_file):
        contact = os.environ.get("WIKIPEDIA_CONTACT")
        self.agent = "history-research-linker/0.1" + (f" ({contact})" if contact else "")
        self.pause, self.last = (0.5 if contact else 6.5), 0.0
        self.path = os.path.join(HERE, "cache", cache_file)
        self.cache = json.load(open(self.path, encoding="utf-8")) if os.path.exists(self.path) else {}

    def get(self, url, params, post=False):
        data = urllib.parse.urlencode(params)
        for _ in range(8):
            time.sleep(max(0.0, self.last + self.pause - time.time()))
            self.last = time.time()
            try:
                req = urllib.request.Request(url if post else url + "?" + data, data.encode() if post else None,
                                             {"User-Agent": self.agent})
                return json.load(urllib.request.urlopen(req, timeout=120))
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(float(e.headers.get("retry-after") or 10) + 1)
        raise RuntimeError(f"{url}: still rate limited; run again, the cache keeps what is done")

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.cache, open(self.path, "w", encoding="utf-8"), ensure_ascii=False)


class Wikipedia(Source):
    def __init__(self):
        super().__init__("wikipedia.json")

    def _query(self, **params):
        return self.get("https://en.wikipedia.org/w/api.php",
                        {"action": "query", "format": "json", "redirects": 1, **params}).get("query", {})

    @staticmethod
    def _entry(page):
        """title and description: the short description and the first sentence, which usually carries the dates"""
        first = re.split(r"(?<=[.!?])\s", page.get("extract", "").strip())[0]
        return {"title": page["title"],
                "description": "; ".join(x for x in (page.get("description", ""), first) if x)[:260]}

    def search(self, query):
        """[{title, description}] of the best full-text matches, disambiguation pages left out"""
        if query not in self.cache:
            pages = self._query(generator="search", gsrsearch=query, gsrlimit=PER_QUERY + 2, exintro=1, explaintext=1,
                                exsentences=2, exlimit="max", prop="description|pageprops|extracts",
                                ppprop="disambiguation").get("pages", {})
            self.cache[query] = [self._entry(p) for p in sorted(pages.values(), key=lambda p: p.get("index", 99))
                                 if "disambiguation" not in p.get("pageprops", {})][:PER_QUERY]
            self.save()
        return self.cache[query]

    def links(self, title):
        """titles of the articles that an article links to"""
        key = "links|" + title
        if key not in self.cache:
            out, more = [], {}
            while more is not None:
                r = self.get("https://en.wikipedia.org/w/api.php",
                             {"action": "query", "format": "json", "redirects": 1, "titles": title, "prop": "links",
                              "plnamespace": 0, "pllimit": "max", **more})
                for p in r.get("query", {}).get("pages", {}).values():
                    out += [x["title"] for x in p.get("links", [])]
                more = {"plcontinue": r["continue"]["plcontinue"]} if "continue" in r else None
            self.cache[key] = out
            self.save()
        return self.cache[key]

    def pages(self, titles):
        """[{title, description}] of known titles, redirects followed, disambiguation pages left out"""
        todo = [t for t in titles if "page|" + t not in self.cache]
        for i in range(0, len(todo), 20):
            got = self._query(titles="|".join(todo[i:i + 20]), exintro=1, explaintext=1, exsentences=2, exlimit="max",
                              prop="description|pageprops|extracts", ppprop="disambiguation")
            final = {r["from"]: r["to"] for r in got.get("redirects", [])}
            by_title = {p["title"]: p for p in got.get("pages", {}).values()}
            for t in todo[i:i + 20]:
                p = by_title.get(final.get(t, t))
                ok = p and "missing" not in p and "disambiguation" not in p.get("pageprops", {})
                self.cache["page|" + t] = self._entry(p) if ok else None
            self.save()
        return [self.cache["page|" + t] for t in titles if self.cache["page|" + t]]


class Yago(Source):
    def __init__(self):
        super().__init__("wikidata.json")
        self.relations = {k: [p.split(":")[1] for p in v["wikidata"]] for k, v in json.load(
            open(os.path.join(HERE, "..", "dataset", "relations.json"), encoding="utf-8")).items()}
        self.schema = self._schema()

    @staticmethod
    def _schema():
        """{Wikidata class: [top classes of the YAGO class it is mapped to]} from yago-schema.ttl"""
        ttl = re.sub(r"#[^\n]*", "", open(os.path.join(HERE, "yago-schema.ttl"), encoding="utf-8").read())
        sup, frm = {}, {}
        for block in re.split(r"\.\s*\n", ttl):
            m = re.match(r"\s*((?:schema|yago):\w+)\s", block)
            if m:
                frm[m.group(1)] = re.findall(r"wd:(Q\d+)", " ".join(re.findall(r"ys:fromClass\s+([^;]+)", block)))
                sup[m.group(1)] = [c for c in re.findall(r"(?:schema|yago):\w+", " ".join(
                    re.findall(r"rdfs:subClassOf\s+([^;]+)", block))) if c != "schema:Thing"]

        def tops(c, seen=()):
            if c in seen:
                return set()
            return {c} if not sup.get(c) else set().union(*[tops(s, seen + (c,)) for s in sup[c]])

        return {q: sorted(tops(c)) for c, qs in frm.items() for q in qs}

    def entities(self, titles):
        """Wikipedia titles -> {title: record}; None if the article has no Wikidata item.
        record: qid, aliases, classes (YAGO top classes; none = not in YAGO), years [first, last] or None,
        facts {Wikidata property: [qid]} for the properties that YAGO's relations are built from"""
        todo = [t for t in dict.fromkeys(titles) if "e|" + t not in self.cache]
        props = sorted({p for ps in self.relations.values() for p in ps})
        for i in range(0, len(todo), 50):
            got = self.get("https://www.wikidata.org/w/api.php", {
                "action": "wbgetentities", "format": "json", "sites": "enwiki", "titles": "|".join(todo[i:i + 50]),
                "props": "claims|labels|aliases|sitelinks", "languages": "en", "sitefilter": "enwiki"})
            found = {}
            for qid, e in got.get("entities", {}).items():
                if "missing" in e:
                    continue
                values = lambda p, kind: [c["mainsnak"]["datavalue"]["value"] for c in e.get("claims", {}).get(p, [])
                                          if c["mainsnak"].get("datavalue", {}).get("type") == kind]
                ids = lambda p: [v["id"] for v in values(p, "wikibase-entityid")]
                years = [int(re.match(r"[+-]\d+", v["time"]).group()) for p in DATES for v in values(p, "time")]
                found[e["sitelinks"]["enwiki"]["title"]] = {
                    "qid": qid, "label": e.get("labels", {}).get("en", {}).get("value"),
                    "aliases": [a["value"] for a in e.get("aliases", {}).get("en", [])],
                    "instance_of": ids("P31"), "years": [min(years), max(years)] if years else None,
                    "facts": {p: ids(p) for p in props if ids(p)}}
            for t in todo[i:i + 50]:
                self.cache["e|" + t] = found.get(t)
            self.save()
        records = {t: self.cache["e|" + t] for t in titles}
        self._classes({c for r in records.values() if r for c in r["instance_of"]})
        for r in records.values():
            if r:
                r["classes"] = sorted({y for c in r["instance_of"] for y in self.cache["c|" + c]})
        return records

    def _classes(self, classes):
        """Under which classes of the schema each Wikidata class lies: one SPARQL query per 150 classes."""
        todo = sorted(c for c in classes if "c|" + c not in self.cache)
        for i in range(0, len(todo), 150):
            query = ("SELECT ?c ?t WHERE { VALUES ?c { %s } VALUES ?t { %s } ?c wdt:P279* ?t }"
                     % (" ".join("wd:" + c for c in todo[i:i + 150]), " ".join("wd:" + t for t in self.schema)))
            rows = self.get("https://query.wikidata.org/sparql", {"query": query, "format": "json"}, post=True)
            under = {}
            for r in rows["results"]["bindings"]:
                under.setdefault(r["c"]["value"].rsplit("/", 1)[1], set()).update(
                    self.schema[r["t"]["value"].rsplit("/", 1)[1]])
            for c in todo[i:i + 150]:
                self.cache["c|" + c] = sorted(under.get(c, ()))
            self.save()

    def has_fact(self, relation, subj, obj):
        """Does YAGO's source state this relation between two records? None if it cannot be checked: a record is
        missing or the relation maps to no Wikidata property (the ex: relations)."""
        props = self.relations.get(relation)
        if not props or not subj or not obj:
            return None
        return any(obj["qid"] in subj["facts"].get(p, []) for p in props)
