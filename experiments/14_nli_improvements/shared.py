"""Shared frozen NLI baseline and canonical archive evaluator; no cached results."""
from __future__ import annotations
import csv, json, sys, importlib.util
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/09_nli_extended'))
sys.path.insert(0,str(ROOT/'experiments/11_nli_error_driven'))
from run import predict, BASE, ALT, MODEL, SYMMETRIC, kind, mentions
from compare_all import convert_nli
from common import score_strategy
from evaluate_reverse import filter_predictions
from reverse_probe import key
DOCS=('alford2026','nechaeva2017')
THRESHOLD=0.90
DELTA=0.20

def read(path):
    path=Path(path)
    if not path.exists():raise FileNotFoundError(f'Missing {path}; run experiments 09/11 first')
    return json.loads(path.read_text(encoding='utf-8'))
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def table(path,rows,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
def evaluator():
    spec=importlib.util.spec_from_file_location('archive_eval_14',ROOT/'archive/eval/evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    return ev

def baseline(source):
    assert source in ('gold','lela')
    raw=read(ROOT/'experiments/09_nli_extended/results'/f'{source}_sentence_sentence_scores.json')
    if not raw or any(len(j.get('hypothesis_scores',[]))<2 for j in raw):raise ValueError('Incomplete NLI score cache')
    selected=predict(score_strategy(raw,'ensemble'),THRESHOLD,False)
    paired=read(ROOT/'experiments/11_nli_error_driven/results/reverse_probe'/f'{source}_ensemble_t0p9_paired_scores.json')
    frozen=filter_predictions(selected,paired,DELTA,'ensemble')
    return raw, frozen, paired

def gold_and_texts():
    ev=evaluator(); gold=ev.load_dir(ROOT/'archive/eval/gold'); texts=ev.load_texts(ROOT/'texts')
    if set(gold)!=set(DOCS):raise ValueError(f'Expected annotated documents {DOCS}, got {list(gold)}')
    return ev,gold,texts

def evaluate(source, raw, selected, ev, gold, texts, bootstrap=100, doc_subset=None):
    docs=tuple(doc_subset or DOCS)
    pred={d:convert_nli(d,raw,selected,gold[d],source,'experiment14') for d in docs}
    res=ev.evaluate({d:texts[d] for d in docs},{d:gold[d] for d in docs},pred,False,{},bootstrap)
    if res['problems']:raise ValueError(f'Evaluator validation: {res["problems"][:3]}')
    f=res['overall']['fact'];pair=res['overall']['fact_pair'];ci=res['ci'].get('fact',(None,None))
    return {'tp':f['tp'],'fp':f['pred']-f['tp'],'pred':f['pred'],'gold':f['gold'], 'precision':f['P'],'recall':f['R'],'f1':f['F1'], 'fact_pair_f1':pair['F1'],'ci_low':ci[0],'ci_high':ci[1]}

def labels(source,raw,selected,ev,gold):
    """Label each selected prediction by the EXACT same entity alignment and relation normalization as evaluator."""
    preds={d:convert_nli(d,raw,selected,gold[d],source,'label') for d in DOCS}
    gold_keys={}
    align={}
    for d in DOCS:
        align[d]=ev.align(gold[d]['entities'],preds[d]['entities'])
        gold_keys[d]={(x['para'],x['subj'],ev.norm(x['rel']),x['obj']) for x in gold[d]['facts']}
    out={}
    for j in selected:
        d=j['doc'];mapped=align[d]
        candidate=(j['para'],mapped.get(j['subj_entity']),ev.norm(j['rel']),mapped.get(j['obj_entity']))
        out[key(j)]=int(candidate in gold_keys[d])
    return out

def report(out,name,source,raw,candidates,conditions,bootstrap=100):
    ev,gold,texts=gold_and_texts();rows=[]
    for label,chosen in conditions.items():
        rows.append({'experiment':name,'source':source,'policy':label,'selected_jobs':len(chosen),**evaluate(source,raw,chosen,ev,gold,texts,bootstrap)})
    table(Path(out)/f'{source}_comparison.csv',rows);return rows
