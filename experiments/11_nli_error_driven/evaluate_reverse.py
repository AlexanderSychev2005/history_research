#!/usr/bin/env python3
"""Apply a *pre-declared* reverse-contrast acceptance rule to original selected predictions.
Forward entailment must exceed reverse entailment by >= delta. This does not
run the model, and never uses gold labels to decide which predictions to keep.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import ROOT,HERE,read,save,inputs,score_strategy,predict,measure,DOCS
from reverse_probe import key,selected_for_probe
from run import SYMMETRIC

def filter_predictions(selected,paired,delta,strategy):
    reverse={tuple(x['key']):x for x in paired}
    out=[]
    for x in selected:
        if x['rel'] in SYMMETRIC:out.append(x);continue # symmetric relation remains untouched
        row=reverse.get(key(x))
        if row is None:raise KeyError(f'Missing reverse counterfactual: {key(x)}')
        rr=row['reverse_scores']
        reverse_score=rr[{'baseline':0,'alternative':1}[strategy]] if strategy!='ensemble' else max(rr)
        if x['score']-reverse_score>=delta:out.append(x)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('gold','lela'),default='gold')
    p.add_argument('--strategy',choices=('baseline','alternative','ensemble'),default='ensemble')
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--deltas',type=float,nargs='+',default=[0.0,0.1,0.2])
    p.add_argument('--scores',type=Path,default=ROOT/'experiments/09_nli_extended/results')
    p.add_argument('--input',type=Path,default=HERE/'results'/'reverse_probe')
    p.add_argument('--out',type=Path,default=HERE/'results'/'reverse_probe'/'comparison.csv')
    a=p.parse_args();stem=f'{a.source}_{a.strategy}_t{str(a.threshold).replace(".","p")}'
    path=a.input/f'{stem}_paired_scores.json'
    if not path.exists():raise FileNotFoundError(f'{path}; first run reverse_probe.py without --prepare-only')
    raw=read(a.scores/f'{a.source}_sentence_sentence_scores.json');paired=read(path)
    selected=predict(score_strategy(raw,a.strategy),a.threshold,False)
    gold,texts,ev=inputs()
    rows=[]
    for name,d in [('original',None)]+[(f'delta_{x:g}',x) for x in a.deltas]:
        kept=selected if d is None else filter_predictions(selected,paired,d,a.strategy)
        stat,_=measure(raw,kept,a.source,gold,texts,ev)
        rows.append({'source':a.source,'strategy':a.strategy,'condition':name,'delta':d,
          'selected':len(kept),**stat})
    save(a.out,rows)
    for r in rows:print(r['condition'],f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}",f"pred={r['pred']}")
    print('Saved',a.out)
if __name__=='__main__':main()
