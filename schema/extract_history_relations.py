#!/usr/bin/env python3
"""YAGO 4.6 SHACL schema -> schema/history_relations.json: the relations used for history, plus three extensions.

  uv run python schema/extract_history_relations.py
"""

import argparse
import json
from pathlib import Path

from rdflib import Graph, Namespace, RDF
from rdflib.collection import Collection

SH = YS = None  # set from the schema's own prefixes in main()

# property -> a short gloss, used as the option description in the relation prompt (subject first, then object)
KEEP = {
    "yago:hasFather": "the object is the father of the subject",
    "yago:hasMother": "the object is the mother of the subject",
    "schema:spouse": "subject and object are married",
    "schema:birthPlace": "the subject was born in the object",
    "schema:deathPlace": "the subject died in the object",
    "yago:burialPlace": "the subject was buried in the object",
    "schema:birthDate": "the subject was born on/in the date",
    "schema:deathDate": "the subject died on/in the date",
    "yago:floruit": "the subject was active at the date",
    "schema:memberOf": "the subject belonged to the group, family, dynasty or institution",
    "schema:worksFor": "the subject served or worked for the object (a ruler, state, army, church)",
    "yago:beliefSystem": "the subject followed the religion or belief system",
    "schema:location": "the subject lived, was based or lies in the place (also 'X of Place' and a place inside a larger place)",
    "yago:creator": "the object wrote or made the subject (a work)",
    "schema:about": "the subject (a work) is about the object",
    "yago:influencedBy": "the subject was influenced by the object",
    "yago:studentOf": "the subject studied under the object",
    "yago:participant": "the object took part in the subject (an event)",
    "yago:leader": "the object led the subject (an event, group or state)",
    "schema:startDate": "the subject began on/in the date",
    "schema:endDate": "the subject ended on/in the date",
    "yago:hasCause": "the subject was caused by the object",
    "yago:winner": "the object won the subject (a battle, war, contest)",
    "yago:neighbor": "the two places border each other",
    "yago:partOf": "the subject is part of the object",
    "yago:follows": "the subject came after (succeeded) the object",
}

EXTENSIONS = {
    "ex:relativeOf": dict(
        domain=["schema:Person"], range=["schema:Person"],
        gloss="subject and object are kin other than parent or spouse (brother, nephew, cousin, in-law)",
        comment="YAGO has only hasFather, hasMother and spouse"),
    "ex:captiveIn": dict(
        domain=["schema:Person", "schema:Organization"], range=["schema:Place", "schema:Organization"],
        gloss="the subject was taken prisoner by, deported to or held captive in the object",
        comment="captivity and deportation; schema:location would lose the coercion"),
    "ex:opposed": dict(
        domain=["schema:Person", "schema:Organization"], range=["schema:Person", "schema:Organization"],
        gloss="the subject fought, rebelled against or persecuted the object",
        comment="conflict between persons, groups and states; YAGO only has Event participants"),
}


def short(g, node):
    s = g.namespace_manager.normalizeUri(node)
    return s.replace("<http://www.w3.org/2001/XMLSchema", "xsd:").rstrip(">")  # xsd: lacks its '#' in this file too


def value_types(g, prop):
    """classes or datatypes a property shape allows, including the alternatives of an sh:or list"""
    out = [short(g, o) for p in (SH["class"], SH.node, SH.datatype) for o in g.objects(prop, p)]
    for lst in g.objects(prop, SH["or"]):
        for alt in Collection(g, lst):
            out += [short(g, o) for p in (SH["class"], SH.node, SH.datatype) for o in g.objects(alt, p)]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="schema/yago-schema.ttl")
    ap.add_argument("--out", default="schema/history_relations.json")
    a = ap.parse_args()

    g = Graph(bind_namespaces="none").parse(a.schema, format="turtle")
    # the downloaded file declares its prefixes without the trailing '#' (sh: <http://www.w3.org/ns/shacl>), so the
    # namespaces are taken from the file itself; 'a' is still the real rdf:type
    global SH, YS
    ns = dict(g.namespaces())
    SH, YS = Namespace(ns["sh"]), Namespace(ns["ys"])
    types = (RDF.type, Namespace(ns["rdf"])["type"])
    shapes = {s for t in types for s in g.subjects(t, SH.NodeShape)}
    rel = {}
    for shape in shapes:
        for prop in g.objects(shape, SH.property):
            path = next(g.objects(prop, SH.path), None)
            name = path and short(g, path)
            if name not in KEEP:
                continue
            r = rel.setdefault(name, dict(gloss=KEEP[name], domain=[], range=[], max_count=None, wikidata=[],
                                          source="yago"))
            r["domain"].append(short(g, shape))
            r["range"] = sorted(set(r["range"]) | set(value_types(g, prop)))
            mc = next(g.objects(prop, SH.maxCount), None)
            if mc is not None:
                r["max_count"] = int(mc)
            r["wikidata"] = sorted(set(r["wikidata"]) | {short(g, w) for w in g.objects(prop, YS.fromProperty)})
    missing = sorted(set(KEEP) - set(rel))
    if missing:
        raise SystemExit(f"not in the schema: {missing}")
    for r in rel.values():
        r["domain"] = sorted(set(r["domain"]))
    for name, e in EXTENSIONS.items():
        rel[name] = dict(gloss=e["gloss"], domain=e["domain"], range=e["range"], max_count=None, wikidata=[],
                         source="extension", comment=e["comment"])

    Path(a.out).write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rel)} relations ({len(EXTENSIONS)} extensions) -> {a.out}")


if __name__ == "__main__":
    main()
