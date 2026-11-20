"""Shared utilities for error-driven NLI diagnostics and reverse-role validation."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/09_nli_extended'))
from run import predict
from compare_all import convert_nli, load_module, fact_rows
DOCS=('alford2026','nechaeva2017')

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def save(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)

def inputs():
    gold={d:read(ROOT/'dataset/gold'/f'{d}.json') for d in DOCS}
    texts={d:{p['id']:p['text'] for p in read(ROOT/'dataset/paragraphs'/f'{d}.json')['paragraphs']} for d in DOCS}
    ev=load_module('history_eval_new_11',ROOT/'archive/eval/evaluate.py')
    return gold,texts,ev

def score_strategy(raw,strategy):
    idx={'baseline':0,'alternative':1,'ensemble':None}[strategy]
    out=[]
    for x in raw:
        if len(x.get('hypothesis_scores',[]))<2:raise ValueError('Missing both hypothesis_scores')
        j=dict(x);ss=x['hypothesis_scores'];j['score']=max(ss) if idx is None else ss[idx];out.append(j)
    return out

def measure(raw,selected,source,gold,texts,ev):
    pred={d:convert_nli(d,raw,selected,gold[d],source,'error_driven') for d in DOCS}
    res,full=fact_rows(ev,texts,gold,pred,1)
    if full['problems']:raise ValueError(str(full['problems'][:3]))
    return res,pred

def normalized_keys(ev,gold,pred,doc):
    """Use precisely evaluator.align and fact_sets semantics, not a hand-built entity match."""
    g=gold[doc];p=pred[doc];al=ev.align(g['entities'],p['entities'])
    gby={}
    for f in g['facts']:
        k=(f['para'],f['subj'],ev.norm(f['rel']),f['obj']);gby[k]=f
    pby={}
    for f in p['facts']:
        k=(f['para'],al.get(f['subj']) or '?'+str(f['subj']),ev.norm(f['rel']),al.get(f['obj']) or '?'+str(f['obj']))
        pby[k]=f
    return gby,pby
