#!/usr/bin/env python3
"""Re-score SAVED NLI entailment probabilities. No inference or gold-based selection.

Run from repository root: python experiments/07_nli_relations/reanalyze_saved.py
Outputs: experiments/07_nli_relations/results/*.csv and *.md
"""
import argparse, csv, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

DOCS = ("alford2026", "nechaeva2017")
THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995)
TYPES = {"person": "schema:Person", "location": "schema:Place",
         "event": "schema:Event", "organization": "schema:Organization",
         "written work": "schema:CreativeWork"}
# Some YAGO constraints refer to non-schema subclasses. These cannot be
# established using our five coarse NER labels; don't incorrectly reject them.
KNOWN_TYPES = set(TYPES.values())
SAME_CLASS = {"yago:partOf", "yago:follows"}

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def options_allowed(option, para_types, schema, mode):
    if mode == "none":
        return True
    rel = option["rel"]
    if rel not in schema:
        return True
    subj_types = para_types.get(option["subj"], set())
    obj_types = para_types.get(option["obj"], set())
    # Missing labels => cannot assert a violation.
    if not subj_types or not obj_types:
        return True
    subj_cls = {TYPES[t] for t in subj_types if t in TYPES}
    obj_cls = {TYPES[t] for t in obj_types if t in TYPES}
    if not subj_cls or not obj_cls:
        return True
    spec = schema[rel]
    def fits(actual, allowed):
        if "schema:Thing" in allowed:
            return True
        if actual & set(allowed):
            return True
        # Unknown subclass could include a match: permit in conservative mode
        if mode == "conservative" and set(allowed) - KNOWN_TYPES:
            return True
        return False
    if not fits(subj_cls, spec["domain"]) or not fits(obj_cls, spec["range"]):
        return False
    if rel in SAME_CLASS and (not subj_cls & obj_cls or subj_cls & obj_cls == {"schema:Person"}):
        return False
    if rel == "yago:partOf" and "schema:Place" in subj_cls & obj_cls:
        return False
    return True

def run(source, base, type_by_para, schema, threshold, mode, pick):
    eid = {e["label"]: e["eid"] for e in base["entities"]}
    facts, seen = [], set()
    for dec in source["decisions"]:
        ptypes = type_by_para.get(dec["paragraph_id"], {})
        opts = [o for o in dec["options"] if options_allowed(o, ptypes, schema, mode)]
        if not opts: continue
        ranked = sorted(opts, key=lambda o:o["p"], reverse=True)
        if pick == "best":
            chosen = ranked[:1]
        elif pick == "all":
            chosen = ranked
        else:
            raise ValueError(pick)
        for o in chosen:
            if o["p"] < threshold: continue
            if o["subj"] not in eid or o["obj"] not in eid: continue
            key=(dec["paragraph_id"], o["subj"], o["rel"], o["obj"])
            if key in seen: continue
            seen.add(key)
            facts.append({"para":key[0],"subj":eid[key[1]],"rel":key[2],"obj":eid[key[3]]})
    return {**base, "facts":facts}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--output",type=Path,default=Path("experiments/07_nli_relations/results"))
    args=ap.parse_args()
    root=args.root.resolve()
    dest=(root / args.output).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("repo_eval",root/"archive/eval/evaluate.py")
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    schema=load(root/"dataset/relations.json")
    texts=ev.load_texts(root/"dataset/paragraphs")
    gold=ev.load_dir(root/"dataset/gold")
    sources={doc:load(root/f"archive/outputs/{doc}_relations_gold__nli_captive.json") for doc in DOCS}
    bases={doc:load(root/f"archive/eval/pred/gold__nli_captive/{doc}.json") for doc in DOCS}
    labels={}
    for doc in DOCS:
        data=load(root/f"archive/outputs/{doc}_ner_gold.json")
        labels[doc]={}
        for paragraph in data["paragraphs"]:
            m=defaultdict(set)
            for mention in paragraph["mentions"]:
                m[mention["text"]].add(mention["label"])
            labels[doc][paragraph["paragraph_id"]]=dict(m)
    rows=[]; best_preds={}
    for mode in ("none","conservative","strict"):
        for pick in ("best","all"):
            for threshold in THRESHOLDS:
                preds={doc:run(sources[doc],bases[doc],labels[doc],schema,threshold,mode,pick) for doc in DOCS}
                result=ev.evaluate(texts,gold,preds,False,{},10)
                d=result["overall"]["fact"]
                rows.append({"mode":mode,"selection":pick,"threshold":threshold,
                             "precision":round(d["P"],6),"recall":round(d["R"],6),
                             "f1":round(d["F1"],6),"tp":d["tp"],"pred":d["pred"],
                             "gold":d["gold"],"pair_f1":round(result["overall"]["fact_pair"]["F1"],6),
                             "issues":len(result["problems"])})
                best_preds[(mode,pick,threshold)]=preds
    with (dest/"ablation_scores.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    # Explicitly report in-sample exploratory maxima, not independent test performance.
    selected = sorted(rows,key=lambda x:(-x["f1"], -x["precision"]))[:12]
    with (dest/"exploratory_top12.md").open("w",encoding="utf-8") as f:
        f.write("# Saved-score NLI ablation: exploratory results\n\n"
                "All configurations are evaluated **on the same 14 gold paragraphs**. "
                "Top rows are **in-sample selections**, not test-set performance. "
                "Only existing `gold__nli_captive` scores are reused, with no NLI model execution.\n\n"
                "Modes: none (raw scores), conservative (type checks only when coarse types permit a decision), "
                "strict (unknown subclass treated as incompatible). "
                "Selection: best (top relation per pair), all (each relation passing threshold).\n\n"
                "| Type filter | Selection | Threshold | P | R | F1 | TP / pred / gold |\n"
                "|---|---|---:|---:|---:|---:|---|\n")
        for row in selected:
            f.write(f"| {row['mode']} | {row['selection']} | {row['threshold']} "
                    f"| {row['precision']:.1%} | {row['recall']:.1%} | {row['f1']:.1%} "
                    f"| {row['tp']}/{row['pred']}/{row['gold']} |\n")
        f.write("\nDo not compare maxima directly with Jev's pre-specified operating point. "
                "For a fair new result, fix type handling and thresholds on independent development data, "
                "then report held-out paragraphs or documents.\n")
    # fixed predeclared illustrative variants; JSON outputs usable by the official evaluator
    fixed=(("raw_best_t099","none","best",0.99),
           ("conservative_best_t099","conservative","best",0.99),
           ("conservative_best_t090","conservative","best",0.9),
           ("conservative_all_t099","conservative","all",0.99))
    for name,mode,pick,threshold in fixed:
        folder=dest/"pred"/name;folder.mkdir(parents=True,exist_ok=True)
        for doc, pred in best_preds[(mode,pick,threshold)].items():
            (folder/f"{doc}.json").write_text(json.dumps(pred,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved {len(rows)} configurations to {dest}")
    for row in selected[:8]:
        print(row["mode"],row["selection"],row["threshold"],
              f"P={row['precision']:.1%} R={row['recall']:.1%} F1={row['f1']:.1%}",
              f"{row['tp']}/{row['pred']}/{row['gold']}")
    for row in rows:
        if row["mode"]=="none" and row["selection"]=="best" and row["threshold"]==0.99:
            print("Raw best at threshold 0.99:",row)

if __name__=="__main__":
    main()
