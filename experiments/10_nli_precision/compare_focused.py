#!/usr/bin/env python3
"""Compare paired original/marked inference with common historical fact evaluator."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/09_nli_extended'))
from run import predict
from compare_all import convert_nli,load_module,fact_rows
DOCS=('alford2026','nechaeva2017')
def load(path):return json.loads(path.read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',choices=('gold','lela'),default='gold')
 p.add_argument('--input',type=Path,default=HERE/'results'/'focused')
 p.add_argument('--threshold',type=float,default=0.9)
 p.add_argument('--out',type=Path,default=HERE/'results'/'focused'/'comparison.csv')
 a=p.parse_args();ev=load_module('history_eval_focused',ROOT/'archive/eval/evaluate.py')
 gold={d:load(ROOT/'dataset/gold'/f'{d}.json') for d in DOCS}
 texts={d:{p['id']:p['text'] for p in load(ROOT/'dataset/paragraphs'/f'{d}.json')['paragraphs']} for d in DOCS}
 rows=[]
 for condition in ('original_matched','entity_marked'):
  path=a.input/f'{a.source}_{condition}_scores.json'
  if not path.exists():raise FileNotFoundError(f'{path} - run focused_inference.py without --prepare-only')
  raw=load(path)
  for strategy in ('baseline','alternative','ensemble'):
   scored=[]
   for j in raw:
    k=dict(j);h=k['hypothesis_scores'];k['score']={'baseline':h[0],'alternative':h[1],'ensemble':max(h)}[strategy];scored.append(k)
   selected=predict(scored,a.threshold,False)
   preds={d:convert_nli(d,raw,selected,gold[d],a.source,condition) for d in DOCS}
   s,result=fact_rows(ev,texts,gold,preds,1)
   rows.append({'source':a.source,'condition':condition,'strategy':strategy,'threshold':a.threshold,**s})
   if result['problems']:raise ValueError(f'Evaluator validation errors: {result["problems"][:2]}')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for row in rows:print(row['condition'],row['strategy'],f"F1={row['f1']:.3f}")
 print('Saved',a.out)
if __name__=='__main__':main()
