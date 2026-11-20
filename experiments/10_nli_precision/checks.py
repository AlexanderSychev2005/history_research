#!/usr/bin/env python3
"""Precision-oriented NLI checks on saved 09_nli_extended scores; no model inference.
Execute from repo root: python experiments/10_nli_precision/checks.py

Uses identical archive/eval/evaluate.py strict paragraph fact matching. All results
are exploratory: existing documents informed model and strategy development.
"""
from __future__ import annotations
import argparse, collections, csv, json, sys, importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
EX9=ROOT/'experiments/09_nli_extended'
sys.path.insert(0,str(EX9))
from run import predict, compatible
from compare_all import convert_nli, load_module, fact_rows
DOCS=('alford2026','nechaeva2017')
MODES=('sentence_sentence','sentence_paragraph','paragraph_paragraph')
SOURCES=('gold','lela')
STRATEGIES=('baseline','alternative','ensemble')

def read(path):return json.loads(path.read_text(encoding='utf8'))
def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def type_index(source,doc,gold):
    if source=='gold':return {e['eid']:e.get('type','') for e in gold['entities']}
    path=ROOT/'lela/pred/llm'/f'{doc}.jsonl'
    if not path.exists():return {}
    payload=json.loads(path.read_text(encoding='utf8').splitlines()[0]);out={}
    from run import norm
    for m in payload['entities']:
        eid=m.get('entity_id') or f"surface:{norm(m['text'])}"
        label=m.get('label','')
        if eid in out and out[eid]!=label:out[eid]='' # conflicting LELA types -> unknown
        else:out[eid]=label
    return out

def typed_filter(jobs,source,gold_by_doc,types,schema,mode):
    if mode=='as_scored':return jobs
    strict=mode=='strict'
    result=[]
    for j in jobs:
        relation=schema.get(j['rel'])
        if relation is None:continue
        mapping=types[(source,j['doc'])]
        subj={'type':mapping.get(j['subj_entity'],'')}
        obj={'type':mapping.get(j['obj_entity'],'')}
        if compatible(subj,relation['domain'],strict) and compatible(obj,relation['range'],strict):result.append(j)
    return result

def score_strategy(jobs,strategy):
    result=[]
    for j in jobs:
        x=dict(j);h=x['hypothesis_scores']
        if len(h)<2:raise ValueError('Requires 2 hypothesis scores per candidate; regenerate ensemble cache')
        x['score']={'baseline':h[0],'alternative':h[1],'ensemble':max(h)}[strategy]
        result.append(x)
    return result

def choose(jobs,threshold,margin,multi=False):
    """Margin = best minus second DISTINCT relation, within same ordered pair.
    For multi-label, margin is intentionally unsupported because ranking ambiguity.
    """
    if margin and multi:raise ValueError('Margin test is single-label only')
    if margin==0:return predict(jobs,threshold,multi)
    groups=collections.defaultdict(dict)
    for j in jobs:
        k=(j['doc'],j['para'],j['subj_entity'],j['obj_entity'])
        r=j['rel'];prev=groups[k].get(r)
        if prev is None or j['score']>prev['score']:groups[k][r]=j
    chosen=[]
    for relmap in groups.values():
        ranked=sorted(relmap.values(),key=lambda x:-x['score'])
        if not ranked or ranked[0]['score']<threshold:continue
        runner_up=ranked[1]['score'] if len(ranked)>1 else 0.0
        if ranked[0]['score']-runner_up>=margin:chosen.append(ranked[0])
    return chosen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,default=EX9/'results')
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--threshold',type=float,default=0.90)
    p.add_argument('--bootstrap',type=int,default=1,help='Use 1 for speed, 500 for uncertainty estimates')
    p.add_argument('--sources',nargs='+',choices=SOURCES,default=list(SOURCES))
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    schema=read(ROOT/'dataset/relations.json')
    gold={d:read(ROOT/'dataset/gold'/f'{d}.json') for d in DOCS}
    texts={d:{x['id']:x['text'] for x in read(ROOT/'dataset/paragraphs'/f'{d}.json')['paragraphs']} for d in DOCS}
    types={(s,d):type_index(s,d,gold[d]) for s in SOURCES for d in DOCS}
    ev=load_module('history_fact_eval_for_precision',ROOT/'archive/eval/evaluate.py')
    rows=[];missing=[];risk=[]
    # Type filter and margins tested at fixed threshold, same cached NLI predictions.
    for source in args.sources:
      for mode in args.modes:
        path=args.scores/f'{source}_{mode}_scores.json'
        if not path.exists():missing.append(str(path));continue
        raw=read(path)
        if not raw or any('hypothesis_scores' not in x for x in raw):raise ValueError(f'Incomplete score cache: {path}')
        for strategy in STRATEGIES:
          scored=score_strategy(raw,strategy)
          for filtering in ('as_scored','conservative','strict'):
            eligible=typed_filter(scored,source,gold,types,schema,filtering)
            for margin in (0.00,0.05,0.10,0.20):
              selected=choose(eligible,args.threshold,margin)
              pred={d:convert_nli(d,raw,selected,gold[d],source,'offline') for d in DOCS}
              s,details=fact_rows(ev,texts,gold,pred,args.bootstrap)
              if args.bootstrap <= 1:
                  s['ci_low']=None; s['ci_high']=None  # one bootstrap draw is not an interval
              row={'source':source,'mode':mode,'strategy':strategy,'filter':filtering,
                   'threshold':args.threshold,'margin':margin,'candidates_before':len(scored),
                   'candidates_after':len(eligible),'selected':len(selected),**s}
              rows.append(row)
              if details['problems']:risk.append({'variant':str(row),'problems':str(details['problems'][:3])})
    write_csv(args.out/'ablations.csv',rows)
    # Cross-document threshold+margin selection: tune only on A, evaluate B, and vice versa.
    # Keep strategy fixed to baseline or ensemble, do not select strategy on test doc.
    cross=[]
    for source in args.sources:
      for mode in args.modes:
        path=args.scores/f'{source}_{mode}_scores.json'
        if not path.exists():continue
        raw=read(path)
        for strategy in ('baseline','ensemble'):
          scored=score_strategy(raw,strategy)
          for filtering in ('as_scored','strict'):
            eligible=typed_filter(scored,source,gold,types,schema,filtering)
            for dev,test in ((DOCS[0],DOCS[1]),(DOCS[1],DOCS[0])):
              candidates=[]
              for threshold in (0.80,0.90,0.95,0.97,0.99):
                for margin in (0.00,0.05,0.10,0.20):
                  selected=choose(eligible,threshold,margin)
                  pred={d:convert_nli(d,raw,selected,gold[d],source,'crossdoc') for d in DOCS}
                  s,_=fact_rows(ev,{dev:texts[dev]},{dev:gold[dev]},{dev:pred[dev]},1)
                  candidates.append((s['f1'],s['precision'],threshold,margin,selected,pred,s))
              # Fixed deterministic tie breaking: higher dev F1, then precision, then larger threshold,
              # then larger margin. No test scores observed during hyperparameter selection.
              best=max(candidates,key=lambda z:z[:4])
              s,_=fact_rows(ev,{test:texts[test]},{test:gold[test]},{test:best[5][test]},1)
              cross.append({'source':source,'mode':mode,'strategy':strategy,'filter':filtering,
                 'development_doc':dev,'evaluation_doc':test,'selected_threshold':best[2],
                 'selected_margin':best[3],'development_f1':best[6]['f1'],
                 'evaluation_precision':s['precision'],'evaluation_recall':s['recall'],
                 'evaluation_f1':s['f1'],'evaluation_tp':s['tp'],'evaluation_pred':s['pred'],
                 'evaluation_gold':s['gold']})
    write_csv(args.out/'cross_document.csv',cross)
    (args.out/'missing_scores.json').write_text(json.dumps(missing,indent=2)+'\n',encoding='utf8')
    (args.out/'validation_issues.json').write_text(json.dumps(risk,indent=2)+'\n',encoding='utf8')
    baseline=[x for x in rows if x['source']=='gold' and x['mode']=='sentence_sentence' and x['strategy']=='ensemble' and x['filter']=='as_scored' and x['margin']==0]
    lines=['# NLI precision-oriented checks (exploratory)','',
      'All rows in `ablations.csv` use **one common archive fact evaluator** and saved NLI scores.',
      'No new model inference was performed by this script. Margins compare the best relation to the next distinct relation for the same directed entity pair.',
      'Strict type constraints are **post-inference**: they can only remove already-scored candidates; they cannot restore candidates rejected during original generation.',
      'Type predictions from LELA can be unknown/inconsistent. Strict filtering may discard valid facts.',
      'Every score is exploratory because both documents were already inspected during development.',
      'Cross-document selection tunes threshold and margin on only one document per fold, but is NOT an untouched test evaluation.',
      '',f'- Offline configurations: {len(rows)}',f'- Cross-document evaluation folds: {len(cross)}',
      f'- Missing score groups: {len(missing)}',f'- Evaluator validation issues: {len(risk)}','']
    if baseline:
      b=baseline[0];lines += [f"Reproduction check (gold/sentence_sentence/ensemble, threshold {args.threshold:.2f}, margin 0): P={b['precision']:.3f}, R={b['recall']:.3f}, F1={b['f1']:.3f}.",'']
    lines += ['## Interpretation rules','',
      'Do not report the maximum F1 over `ablations.csv` as test performance. Compare a **preselected** configuration or use held-out documents.',
      'A larger margin may reduce recall: inspect both precision and recall and candidate coverage.',
      'The `cross_document.csv` results estimate transfer between two already-seen articles and have high uncertainty.','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print(f'Finished offline NLI precision checks: {len(rows)} configurations, {len(cross)} cross-document folds, {len(missing)} missing caches, {len(risk)} evaluator issues')
    if baseline:print('Baseline reference F1:',round(baseline[0]['f1'],4))

if __name__=='__main__':main()
