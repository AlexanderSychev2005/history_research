#!/usr/bin/env python3
"""Builds eval/gold/<doc>.json (and the gold mentions as outputs/<doc>_ner_gold.json) from the tables below.
Conventions: eval/gold/README.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from relations_jev import sentences  # the same sentence split the extractor uses

# eid -> (label, type). Surface forms found in the text are added as aliases automatically. States are 'location':
# YAGO types historical states (Sasanian Empire, Byzantine Empire) as schema:Country, a schema:Place.
ENTITIES = {
    "alford2026": {
        "REB": ("Armenian rebellion", "event"), "YII": ("Yazdgerd II", "person"), "AWA": ("Awarayr", "location"),
        "VAR": ("Vardan Mamikonean", "person"), "ARC": ("Armenian Church", "organization"),
        "LAZ": ("Łazar P‘arpec‘i", "person"), "HOA": ("History of the Armenians", "written work"),
        "GOR": ("Gorgān", "location"), "ALB": ("Alborz", "location"), "CAS": ("Caspian Sea", "location"),
        "KHU": ("Khuzestan", "location"), "VAH": ("Vahan Mamikonean", "person"), "SAS": ("Sasanian Empire", "location"),
        "VAL": ("Vałaršapat", "location"), "ELI": ("Ełišē", "person"), "THO": ("Robert Thomson", "person"),
        "AKI": ("Akinean", "person"), "KIW": ("Kiwlēsērean", "person"), "PHI": ("Philo", "person"),
        "JOE": ("John of Ephesus", "person"), "ECH": ("Ecclesiastical Histories", "written work"),
        "NER": ("Nerseh Kamsarakan", "person"), "SIR": ("Širak", "location"), "ARS": ("Aršawir Kamsarakan", "person"),
        "HRA": ("Hrahat", "person"), "AKE": ("Battle of Akesga", "event"), "ARM": ("Aršamunik‘", "location"),
        "ARN": ("Armenia", "location"), "KID": ("Kidarite Huns", "organization"), "PER": ("Pērōz", "person"),
        "HEP": ("Hephthalite Huns", "organization"), "TOK": ("Tokharistan", "location"), "HIN": ("Hindu Kush", "location"),
        "VSK": ("Vasak", "person"), "ART": ("Artašēs", "person"), "KAM": ("Kamsarakan family", "organization"),
        "TAT": ("T‘at‘ik", "person"), "BAS": ("Basean", "location"), "VSI": ("Vasak Siwni", "person"),
        "SWK": ("Siwnik‘", "location"), "MAM": ("Mamikoneans", "organization"), "ASU": ("Ašuša", "person"),
        "GUG": ("Gugark‘", "location"), "ASO": ("Asorestan", "location"), "LEW": ("Łewondeank‘", "organization"),
    },
    "nechaeva2017": {
        "AGA": ("Agathias", "person"), "DAM": ("Damascius", "person"), "SYR": ("Syria", "location"),
        "SIM": ("Simplicius", "person"), "CIL": ("Cilicia", "location"), "EUL": ("Eulamius", "person"),
        "PHR": ("Phrygia", "location"), "PRI": ("Priscian", "person"), "LYD": ("Lydia", "location"),
        "HER": ("Hermes", "person"), "DIO": ("Diogenes", "person"), "PHO": ("Phoenicia", "location"),
        "ISI": ("Isidore", "person"), "GAZ": ("Gaza", "location"), "PERS": ("Sasanian Empire", "location"),
        "ROM": ("Byzantine Empire", "location"), "PLA": ("Plato", "person"), "DOM": ("Domitziolus", "person"),
        "BUZ": ("Buzes", "person"), "CAL": ("Callinicum", "location"), "IZD": ("Izdegerd", "person"),
        "HOR": ("Hormizd", "person"), "ARZ": ("Arzanene", "location"), "PRM": ("Persarmenia", "location"),
        "PSZ": ("Pseudo-Zacharias", "person"), "PRO": ("Procopius", "person"), "DAG": ("Dagaris", "person"),
        "SIT": ("Sittas", "person"), "DOR": ("Dorotheus", "person"), "ARN": ("Armenia", "location"),
        "HUN": ("Huns", "organization"), "GRI": ("Grigor", "person"), "JUS": ("Justinian", "person"),
        "MART": ("Martyrdom of Mar Grigor", "written work"), "CHO": ("Chosroes", "person"),
        "ZAB": ("Zabergān", "person"), "COE": ("Church of the East", "organization"),
        "CON": ("Constantinople", "location"),
    },
}

# (paragraph, context that occurs once in it or None, mention inside the context, eid). In paragraph order.
MENTIONS = {
    "alford2026": [
        ("p001", None, "Armenian rebellion", "REB"), ("p001", None, "Yazdgerd II", "YII"),
        ("p003", None, "Armenian rebellion", "REB"), ("p003", None, "Awarayr", "AWA"),
        ("p003", None, "Yazdgerd II", "YII"), ("p003", None, "Vardan Mamikonean", "VAR"),
        ("p003", None, "Armenian Church", "ARC"),
        ("p004", None, "Łazar P‘arpec‘i", "LAZ"), ("p004", None, "History of the Armenians", "HOA"),
        ("p004", None, "Gorgān", "GOR"), ("p004", None, "Hyrcania", "GOR"), ("p004", None, "Alborz", "ALB"),
        ("p004", None, "Caspian Sea", "CAS"), ("p004", "children held in Khuzestan", "Khuzestan", "KHU"),
        ("p004", "Łazar is a", "Łazar", "LAZ"), ("p004", None, "Vahan Mamikonean", "VAH"),
        ("p004", "captives of Khuzestan", "Khuzestan", "KHU"), ("p004", None, "Sasanians", "SAS"),
        ("p004", "Łazar had been", "Łazar", "LAZ"), ("p004", "been Vahan’s", "Vahan", "VAH"),
        ("p004", None, "Vałaršapat", "VAL"), ("p004", None, "Eǰmiacin", "VAL"), ("p004", "by Vahan c.", "Vahan", "VAH"),
        ("p005", "rebellion, Ełišē,", "Ełišē", "ELI"), ("p005", None, "Gorgān", "GOR"), ("p005", None, "Łazar", "LAZ"),
        ("p005", "Ełišē’s dating", "Ełišē", "ELI"), ("p005", None, "Robert Thomson", "THO"),
        ("p005", None, "Akinean", "AKI"), ("p005", None, "Kiwlēsērean", "KIW"), ("p005", "that Ełišē wrote", "Ełišē", "ELI"),
        ("p005", None, "Philo", "PHI"), ("p005", None, "John of Ephesus", "JOE"),
        ("p005", None, "Ecclesiastical Histories", "ECH"),
        ("p013", None, "Gorgān", "GOR"), ("p013", None, "Ełišē", "ELI"), ("p013", None, "Nerseh Kamsarakan", "NER"),
        ("p013", "lord Nerseh of", "Nerseh", "NER"), ("p013", None, "Širak", "SIR"),
        ("p013", None, "Aršawir Kamsarakan", "ARS"), ("p013", "this Nerseh’s", "Nerseh", "NER"),
        ("p013", "from Łazar’s list", "Łazar", "LAZ"), ("p013", "since Łazar had", "Łazar", "LAZ"),
        ("p013", "relationship with Nerseh.", "Nerseh", "NER"), ("p013", "Łazar lived", "Łazar", "LAZ"),
        ("p013", "with Nerseh and", "Nerseh", "NER"), ("p013", None, "Hrahat", "HRA"), ("p013", None, "Vałaršapat", "VAL"),
        ("p013", "cited Nerseh", "Nerseh", "NER"), ("p013", None, "Battle of Akesga", "AKE"),
        ("p013", "Łazar elsewhere", "Łazar", "LAZ"), ("p013", "of Nerseh from", "Nerseh", "NER"),
        ("p013", None, "Aršamunik‘", "ARM"), ("p013", None, "Armenia", "ARN"), ("p013", "that Nerseh spent", "Nerseh", "NER"),
        ("p018", "between the Sasanians", "Sasanians", "SAS"), ("p018", None, "Kidarite Huns", "KID"),
        ("p018", "after Yazdgerd and", "Yazdgerd", "YII"), ("p018", None, "Pērōz", "PER"),
        ("p018", "paying off the Kidarites", "Kidarites", "KID"), ("p018", "the Kidarites’ destruction", "Kidarites", "KID"),
        ("p018", None, "Hephthalite Huns", "HEP"), ("p018", "Yazdgerd paired", "Yazdgerd", "YII"),
        ("p018", "region as the Kidarites", "Kidarites", "KID"), ("p018", None, "Tokharistan", "TOK"),
        ("p018", None, "Hindu Kush", "HIN"),
        ("p024", None, "Khuzestan", "KHU"), ("p024", None, "Vahan Mamikonean", "VAH"),
        ("p024", "brothers Vasak and", "Vasak", "VSK"), ("p024", None, "Artašēs", "ART"),
        ("p024", None, "Kamsarakan family", "KAM"), ("p024", None, "T‘at‘ik", "TAT"), ("p024", None, "Basean", "BAS"),
        ("p024", "Łazar claims they", "Łazar", "LAZ"), ("p024", None, "Vasak Siwni", "VSI"), ("p024", None, "Siwnik‘", "SWK"),
        ("p024", "Still, Łazar", "Łazar", "LAZ"), ("p024", "than Vahan and", "Vahan", "VAH"),
        ("p025", None, "Mamikoneans", "MAM"), ("p025", "named Ašuša", "Ašuša", "ASU"), ("p025", None, "Gugark‘", "GUG"),
        ("p025", None, "Łazar P‘arpec‘i", "LAZ"), ("p025", "Ašuša himself", "Ašuša", "ASU"), ("p025", None, "Persia", "SAS"),
        ("p032", None, "Asorestan", "ASO"), ("p032", "– T‘at‘ik of", "T‘at‘ik", "TAT"), ("p032", None, "Basean", "BAS"),
        ("p032", None, "Łewondeank‘", "LEW"), ("p032", None, "Khuzestan", "KHU"), ("p032", "that T‘at‘ik’s", "T‘at‘ik", "TAT"),
    ],
    "nechaeva2017": [
        ("p004", "According to Agathias", "Agathias", "AGA"), ("p004", None, "Damascius", "DAM"),
        ("p004", None, "Syria", "SYR"), ("p004", None, "Simplicius", "SIM"), ("p004", None, "Cilicia", "CIL"),
        ("p004", None, "Eulamius", "EUL"), ("p004", None, "Phrygia", "PHR"), ("p004", None, "Priscian", "PRI"),
        ("p004", None, "Lydia", "LYD"), ("p004", None, "Hermes", "HER"), ("p004", None, "Diogenes", "DIO"),
        ("p004", None, "Phoenicia", "PHO"), ("p004", None, "Isidore", "ISI"), ("p004", None, "Gaza", "GAZ"),
        ("p004", "moved to Persia “because", "Persia", "PERS"), ("p004", None, "Romans", "ROM"),
        ("p004", None, "Plato", "PLA"), ("p004", "moved to Persia with", "Persia", "PERS"),
        ("p004", "Agathias tells", "Agathias", "AGA"),
        ("p013", "commander, Domitziolus", "Domitziolus", "DOM"), ("p013", None, "Domnentiolus", "DOM"),
        ("p013", None, "Buzes", "BUZ"), ("p013", "by the Persians", "Persians", "PERS"), ("p013", None, "Callinicum", "CAL"),
        ("p013", "nobleman, Izdegerd", "Izdegerd", "IZD"), ("p013", None, "Hormizd", "HOR"), ("p013", None, "Arzanene", "ARZ"),
        ("p013", None, "Persarmenia", "PRM"), ("p013", None, "Romans", "ROM"),
        ("p013", "Pseudo-Zacharias’ description", "Pseudo-Zacharias", "PSZ"),
        ("p013", "capture of Domitziolus", "Domitziolus", "DOM"), ("p013", "capture of Izdegerd", "Izdegerd", "IZD"),
        ("p013", "and Pseudo-Zacharias is", "Pseudo-Zacharias", "PSZ"), ("p013", "mentions Izdegerd", "Izdegerd", "IZD"),
        ("p014", "described by Procopius", "Procopius", "PRO"), ("p014", "that Dagaris", "Dagaris", "DAG"),
        ("p014", None, "Sittas", "SIT"), ("p014", None, "Dorotheus", "DOR"), ("p014", None, "Armenia", "ARN"),
        ("p014", "allies of Persia", "Persia", "PERS"), ("p014", None, "Romans", "ROM"),
        ("p014", "the Persians received", "Persians", "PERS"), ("p014", "Procopius characterizes", "Procopius", "PRO"),
        ("p014", "returned Dagaris", "Dagaris", "DAG"), ("p014", "invading Huns", "Huns", "HUN"),
        ("p018", None, "Pīrān-Gušnasp", "GRI"), ("p018", "name Grigor", "Grigor", "GRI"),
        ("p018", "persecution: Grigor’s", "Grigor", "GRI"), ("p018", "imprisonment, Grigor", "Grigor", "GRI"),
        ("p018", "charge, Grigor", "Grigor", "GRI"), ("p018", None, "Justinian", "JUS"),
        ("p018", "position in Persia", "Persia", "PERS"), ("p018", "than Grigor had", "Grigor", "GRI"),
        ("p018", "had held in Persia", "Persia", "PERS"), ("p018", "appointment of Grigor", "Grigor", "GRI"),
        ("p018", None, "Roman Empire", "ROM"),
        ("p019", None, "Martyrium", "MART"), ("p019", None, "Chosroes", "CHO"), ("p019", "envoy, Zabergān", "Zabergān", "ZAB"),
        ("p019", None, "Mar Grigor", "GRI"), ("p019", "concern that Grigor", "Grigor", "GRI"),
        ("p019", "his faith in Persia", "Persia", "PERS"), ("p019", "the Martyrdom, however", "Martyrdom", "MART"),
        ("p019", "Grigor himself", "Grigor", "GRI"), ("p019", None, "Church of the East", "COE"),
        ("p019", None, "Constantinople", "CON"), ("p019", "by the Martyrdom", "Martyrdom", "MART"),
        ("p019", "promise that Grigor", "Grigor", "GRI"), ("p019", "Zabergān agreed", "Zabergān", "ZAB"),
    ],
}

# (paragraph, subject, relation, object): stated or directly implied in one sentence that names both.
# Symmetric relations (spouse, neighbor, relativeOf) have the name that comes first in the text as subject.
FACTS = {
    "alford2026": [
        ("p003", "REB", "schema:location", "AWA"),
        ("p004", "HOA", "yago:creator", "LAZ"), ("p004", "GOR", "yago:neighbor", "CAS"),
        ("p004", "VAH", "ex:captiveIn", "KHU"), ("p004", "VAH", "ex:opposed", "SAS"),
        ("p004", "LAZ", "schema:location", "VAL"),
        ("p005", "ECH", "yago:creator", "JOE"),
        ("p013", "NER", "ex:captiveIn", "GOR"), ("p013", "ARS", "ex:captiveIn", "GOR"),
        ("p013", "NER", "yago:hasFather", "ARS"), ("p013", "NER", "schema:location", "SIR"),
        ("p013", "NER", "ex:relativeOf", "HRA"), ("p013", "LAZ", "schema:location", "VAL"),
        ("p013", "ARM", "schema:location", "ARN"),
        ("p018", "SAS", "ex:opposed", "KID"), ("p018", "HEP", "yago:follows", "KID"),
        ("p024", "VAH", "ex:captiveIn", "KHU"), ("p024", "VSK", "ex:captiveIn", "KHU"),
        ("p024", "ART", "ex:captiveIn", "KHU"), ("p024", "TAT", "ex:captiveIn", "KHU"),
        ("p024", "VAH", "ex:relativeOf", "VSK"), ("p024", "VAH", "ex:relativeOf", "ART"),
        ("p024", "VSK", "ex:relativeOf", "ART"), ("p024", "TAT", "schema:location", "BAS"),
        ("p025", "ASU", "schema:location", "GUG"), ("p025", "ASU", "ex:captiveIn", "SAS"),
        ("p032", "TAT", "schema:deathPlace", "ASO"), ("p032", "TAT", "schema:location", "BAS"),
    ],
    "nechaeva2017": [
        ("p004", "DAM", "schema:location", "SYR"), ("p004", "SIM", "schema:location", "CIL"),
        ("p004", "EUL", "schema:location", "PHR"), ("p004", "PRI", "schema:location", "LYD"),
        ("p004", "HER", "schema:location", "PHO"), ("p004", "DIO", "schema:location", "PHO"),
        ("p004", "ISI", "schema:location", "GAZ"),
        *[("p004", x, "schema:location", "PERS") for x in ("DAM", "SIM", "EUL", "PRI", "HER", "DIO", "ISI")],
        ("p013", "DOM", "ex:relativeOf", "BUZ"), ("p013", "DOM", "ex:captiveIn", "PERS"),
        ("p013", "IZD", "ex:relativeOf", "HOR"), ("p013", "ARZ", "schema:location", "PRM"),
        ("p013", "IZD", "ex:captiveIn", "ROM"),
        ("p014", "DAG", "schema:location", "ARN"), ("p014", "DAG", "ex:opposed", "HUN"),
        ("p018", "GRI", "schema:worksFor", "JUS"), ("p018", "GRI", "schema:location", "PERS"),
        ("p018", "GRI", "schema:worksFor", "ROM"),
        ("p019", "GRI", "schema:memberOf", "COE"), ("p019", "GRI", "schema:location", "CON"),
    ],
}


def main():
    out = Path("eval/gold")
    out.mkdir(parents=True, exist_ok=True)
    for doc, ents in ENTITIES.items():
        paras = {p["id"]: p["text"] for p in json.loads(Path(f"texts/{doc}.json").read_text(encoding="utf-8"))["paragraphs"]}
        mentions, aliases = [], {e: set() for e in ents}
        for para, ctx, name, eid in MENTIONS[doc]:
            text = paras[para]
            ctx = ctx or name
            assert text.count(ctx) == 1, f"{doc} {para}: {ctx!r} occurs {text.count(ctx)} times"
            assert name in ctx, (doc, para, ctx, name)
            start = text.index(ctx) + ctx.index(name)
            mentions.append({"para": para, "start": start, "end": start + len(name), "text": name, "entity": eid})
            aliases[eid].add(name)
        for para, s, r, o in FACTS[doc]:
            together = any({s, o} <= {m["entity"] for m in mentions if m["para"] == para and a <= m["start"] < b}
                           for a, b in sentences(paras[para]))
            assert together, f"{doc} {para}: no sentence names both {s} and {o} ({r})"
        gold = {
            "doc_id": doc,
            "annotator": "claude-draft (from the text only, before seeing relation output); needs historian review",
            "scope": {"paragraphs": sorted({m["para"] for m in mentions})},
            "entities": [{"eid": e, "label": lab, "type": typ, "aliases": sorted(aliases[e] - {lab})}
                         for e, (lab, typ) in ents.items()],
            "mentions": sorted(mentions, key=lambda m: (m["para"], m["start"])),
            "facts": [{"para": p, "subj": s, "rel": r, "obj": o} for p, s, r, o in FACTS[doc]],
        }
        (out / f"{doc}.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
        kind = {e: typ for e, (_, typ) in ents.items()}
        ner = {"doc_id": doc, "paragraphs": [
            {"paragraph_id": pid, "text": paras[pid],
             "mentions": [{"text": m["text"], "start": m["start"], "end": m["end"], "label": kind[m["entity"]],
                           "score": 1.0} for m in gold["mentions"] if m["para"] == pid]}
            for pid in gold["scope"]["paragraphs"]]}
        Path(f"outputs/{doc}_ner_gold.json").write_text(json.dumps(ner, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{doc}: {len(gold['scope']['paragraphs'])} paragraphs, {len(mentions)} mentions, "
              f"{len(gold['facts'])} facts -> {out / (doc + '.json')}")


if __name__ == "__main__":
    main()
