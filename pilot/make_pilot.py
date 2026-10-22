"""Pilot for testing LELA: one paragraph from each paper plus a basic gold set.

Run from the project root:  python pilot/make_pilot.py
  texts/<doc>.json  ->  pilot/texts/<doc>.json   (the two paragraphs, unchanged)
                        pilot/lela_in/<doc>.txt  (the same paragraph as plain text, the input of LELA)
                        pilot/gold.json          (mentions with offsets + the YAGO entity they should link to)

Gold = proper names of persons, places, peoples/states and works, flat spans, name only (no titles, no "of X"
epithets, no adjectives). `entity` is a YAGO 4.5 id, or None when YAGO has no entity for the thing: the History
of the Armenians, Hermes and Isidore of Gaza (no item), and the region Syria (YAGO has only the modern country,
Roman_Syria and Greater_Syria, none of which is meant). All ids were looked up in the YAGO 4.5 dump.
"""
import json
import os

PARAGRAPH = {"alford2026": "p004", "nechaeva2017": "p004"}

# (doc, context that occurs once in the paragraph, mention inside it, entity)
G = [
    ("alford2026", "historian Łazar P‘arpec‘i includes", "Łazar P‘arpec‘i", "Ghazar Parpetsi"),
    ("alford2026", "in his History of the Armenians,", "History of the Armenians", None),
    ("alford2026", "sent to Gorgān (Latin", "Gorgān", "Hyrcania"),
    ("alford2026", "(Latin. Hyrcania, the region", "Hyrcania", "Hyrcania"),
    ("alford2026", "north of the Alborz watershed", "Alborz", "Alborz"),
    ("alford2026", "bordering the Caspian Sea", "Caspian Sea", "Caspian Sea"),
    ("alford2026", "children held in Khuzestan,", "Khuzestan", "Khuzestan province"),
    ("alford2026", "Łazar is a particularly", "Łazar", "Ghazar Parpetsi"),
    ("alford2026", "sponsored by Vahan Mamikonean", "Vahan Mamikonean", "Vahan I Mamikonian"),
    ("alford2026", "captives of Khuzestan and", "Khuzestan", "Khuzestan province"),
    ("alford2026", "against the Sasanians from", "Sasanians", "Sasanian Empire"),
    ("alford2026", "Łazar had been Vahan’s", "Łazar", "Ghazar Parpetsi"),
    ("alford2026", "been Vahan’s childhood", "Vahan", "Vahan I Mamikonian"),
    ("alford2026", "monastery of Vałaršapat (modern", "Vałaršapat", "Vagharshapat"),
    ("alford2026", "(modern Eǰmiacin)", "Eǰmiacin", "Vagharshapat"),
    ("alford2026", "by Vahan c. 493", "Vahan", "Vahan I Mamikonian"),
    ("nechaeva2017", "According to Agathias, the seven", "Agathias", "Agathias"),
    ("nechaeva2017", "—Damascius of Syria,", "Damascius", "Damascius"),
    ("nechaeva2017", "Damascius of Syria,", "Syria", None),
    ("nechaeva2017", "Simplicius of Cilicia,", "Simplicius", "Simplicius of Cilicia"),
    ("nechaeva2017", "Simplicius of Cilicia,", "Cilicia", "Cilicia"),
    ("nechaeva2017", "Eulamius of Phrygia,", "Eulamius", "Eulamius"),
    ("nechaeva2017", "Eulamius of Phrygia,", "Phrygia", "Phrygia"),
    ("nechaeva2017", "Priscian of Lydia,", "Priscian", "Priscian of Lydia"),
    ("nechaeva2017", "Priscian of Lydia,", "Lydia", "Lydia"),
    ("nechaeva2017", "Hermes and Diogenes", "Hermes", None),
    ("nechaeva2017", "and Diogenes of Phoenicia,", "Diogenes", "Diogenes of Phoenicia"),
    ("nechaeva2017", "Diogenes of Phoenicia,", "Phoenicia", "Phoenicia"),
    ("nechaeva2017", "and Isidore of Gaza—", "Isidore", None),
    ("nechaeva2017", "Isidore of Gaza—", "Gaza", "Gaza City"),
    ("nechaeva2017", "moved to Persia “because", "Persia", "Sasanian Empire"),
    ("nechaeva2017", "prevailing among the Romans", "Romans", "Byzantine Empire"),
    ("nechaeva2017", "“Plato’s philosopher king”", "Plato", "Plato"),
    ("nechaeva2017", "have moved to Persia with", "Persia", "Sasanian Empire"),
    ("nechaeva2017", "Agathias tells us", "Agathias", "Agathias"),
]


def main():
    os.makedirs("pilot/texts", exist_ok=True)
    os.makedirs("pilot/lela_in", exist_ok=True)
    paras, mentions = {}, []
    for doc, pid in PARAGRAPH.items():
        doc_paras = json.load(open(f"texts/{doc}.json", encoding="utf-8"))["paragraphs"]
        paras[doc] = next(p["text"] for p in doc_paras if p["id"] == pid)
        json.dump({"doc_id": doc, "paragraphs": [{"id": pid, "text": paras[doc]}]},
                  open(f"pilot/texts/{doc}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        open(f"pilot/lela_in/{doc}.txt", "w", encoding="utf-8").write(paras[doc])    # LELA reads one plain-text file
    for doc, ctx, surface, title in G:
        text = paras[doc]
        assert text.count(ctx) == 1, f"context not unique: {ctx!r}"
        start = text.index(ctx) + ctx.index(surface)
        mentions.append({"doc": doc, "para": PARAGRAPH[doc], "start": start, "end": start + len(surface),
                         "text": surface, "entity": f"yago:{title.replace(' ', '_')}" if title else None})
    mentions.sort(key=lambda m: (m["doc"], m["start"]))
    json.dump({"mentions": mentions}, open("pilot/gold.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for m in mentions:
        assert paras[m["doc"]][m["start"]:m["end"]] == m["text"]
    print(len(mentions), "mentions;", sum(m["entity"] is None for m in mentions), "without entity")


if __name__ == "__main__":
    main()
