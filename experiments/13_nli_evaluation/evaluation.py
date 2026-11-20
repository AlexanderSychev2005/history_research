#!/usr/bin/env python3
"""Evaluate frozen NLI with the same archive/eval/evaluate.py protocol used by
experiments/08_constrained_relations/summarize.py (shared 14-paragraph gold).
All metrics are exploratory: model and delta were tuned on this very gold set.
Run from anywhere: python experiments/13_nli_evaluation/evaluation.py
"""
from __future__ import annotations
import argparse, collections, csv, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/09_nli_extended'))
sys.path.insert(0, str(ROOT/'experiments/11_nli_error_driven'))
from run import predict
from compare_all import convert_nli
from common import score_strategy
from evaluate_reverse import filter_predictions
DOCS=('alford2026','nechaeva2017')
REFS={
 'zero_shot_sentence_r1': 'archive/experiments/08_constrained_relations/pred/gold__llm-20b-sentence-r1',
 'zero_shot_paragraph_r1': 'archive/experiments/08_constrained_relations/pred/gold__llm-20b-paragraph-r1',
 'zero_shot_paragraph_r2': 'archive/experiments/08_constrained_relations/pred/gold__llm-20b-paragraph-r2',
 'jev_api_gold': 'archive/eval/pred/gold__jev-api',
 'jev_api_guided_gold': 'archive/eval/pred/gold__jev-api-guided',
 'openjev_gold': 'archive/eval/pred/gold__openjev',
}

def load(p): return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p, x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def write_csv(p, rows, fields=None):
 p.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for x in rows for k in x))
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_evaluator():
 spec=importlib.util.spec_from_file_location('archive_eval_13',ROOT/'archive/eval/evaluate.py')
 ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev);return ev

def selected_variants(source,threshold,delta):
 raw=load(ROOT/'experiments/09_nli_extended/results'/f'{source}_sentence_sentence_scores.json')
 original=predict(score_strategy(raw,'ensemble'),threshold,False)
 stem=f'{source}_ensemble_t{str(threshold).replace(".","p")}'
 paired=load(ROOT/'experiments/11_nli_error_driven/results/reverse_probe'/f'{stem}_paired_scores.json')
 reverse=filter_predictions(original,paired,delta,'ensemble')
 return raw,original,reverse

def by_relation_and_errors(ev,gold,preds,run):
 totals=collections.defaultdict(lambda:collections.Counter())
 errors=[]
 for doc in DOCS:
  g=gold[doc];p=preds[doc]; al=ev.align(g['entities'],p['entities'])
  for para in g['scope']['paragraphs']:
   gg=[f for f in g['facts'] if f['para']==para]
   pp=[f for f in p['facts'] if f.get('para')==para]
   gs,_=ev.fact_sets(gg,lambda e:('e',e),{})
   ps,_=ev.fact_sets(pp,lambda e:('e',al[e]) if al.get(e) else ('?',e),{})
   for rel in {x[1] for x in gs|ps}:
    gset={x for x in gs if x[1]==rel};pset={x for x in ps if x[1]==rel};c=totals[rel]
    c['tp']+=len(gset & pset);c['fp']+=len(pset-gset);c['fn']+=len(gset-pset)
   for status,facts in [('FP',ps-gs),('FN',gs-ps)]:
    for subj,rel,obj in sorted(facts,key=str):
     def pretty(e):
      if e[0]=='lit':return e[1]
      if e[0]=='e':return next((x['label'] for x in g['entities'] if x['eid']==e[1]),e[1])
      return str(e[1])
     errors.append({'run':run,'doc':doc,'para':para,'type':status,'subject':pretty(subj),'relation':rel,'object':pretty(obj)})
 rows=[]
 for rel,c in sorted(totals.items()):
  tp,fp,fn=c['tp'],c['fp'],c['fn'];P,R,F1=ev.prf(tp,tp+fp,tp+fn)
  rows.append({'run':run,'relation':rel,'tp':tp,'fp':fp,'missed':fn,'P':P,'R':R,'F1':F1})
 return rows,errors

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--threshold',type=float,default=.9)
 ap.add_argument('--delta',type=float,default=.2)
 ap.add_argument('--bootstrap',type=int,default=500)
 ap.add_argument('--out',type=Path,default=HERE/'results')
 a=ap.parse_args(argv)
 if a.threshold!=.9 or a.delta!=.2:raise ValueError('Frozen experiment: threshold=0.9, delta=0.2. Change code & mark new exploration if tuning.')
 ev=load_evaluator();out=a.out
 # Exactly the files the constrained LLM report uses; NOT dataset/gold or a reformatted surrogate.
 gold=ev.load_dir(ROOT/'archive/eval/gold')
 texts=ev.load_texts(ROOT/'texts')
 assert set(gold)==set(DOCS)
 rows=[];relrows=[];errors=[];validation=[]
 preds={}
 for source in ('gold','lela'):
  raw,original,reverse=selected_variants(source,a.threshold,a.delta)
  for mode,chosen in [('original',original),('reverse_d0p2',reverse)]:
   name=f'nli_{source}_{mode}'
   pred={d:convert_nli(d,raw,chosen,gold[d],source,name) for d in DOCS}
   preds[name]=pred
   for d in DOCS:dump(out/'pred'/name/f'{d}.json',pred[d])
 # Reference predictions evaluated by the same archive scorer
 for name,relative in REFS.items():
  directory=ROOT/relative
  if not all((directory/f'{d}.json').exists() for d in DOCS):
   print(f'SKIP missing reference {name}: {directory}',file=sys.stderr);continue
  preds[name]={d:load(directory/f'{d}.json') for d in DOCS}
 for name,pred in preds.items():
  result=ev.evaluate(texts,gold,pred,False,{},a.bootstrap) # exact overlap setting from teammate
  stats=result['overall'];ci=result['ci']
  f=stats['fact'];pair=stats['fact_pair'];lo,hi=ci.get('fact',(None,None))
  rows.append({'run':name,'condition':'gold' if name.startswith('nli_gold') else 'lela' if name.startswith('nli_lela') else 'reference_gold',
    'paragraphs':result['paragraphs'],'tp':f['tp'],'pred':f['pred'],'gold':f['gold'],'P':f['P'],'R':f['R'],'F1':f['F1'],
    'ci_low':lo,'ci_high':hi,'fact_pair_tp':pair['tp'],'fact_pair_pred':pair['pred'],'fact_pair_gold':pair['gold'],
    'fact_pair_P':pair['P'],'fact_pair_R':pair['R'],'fact_pair_F1':pair['F1'],'issues':len(result['problems'])})
  validation.extend({'run':name,'issue':x} for x in result['problems'])
  r,e=by_relation_and_errors(ev,gold,pred,name);relrows.extend(r);errors.extend(e)
  if name.startswith('nli_'):
   for d in DOCS:
    check=ev.check(pred[d],texts,d)[0]
    if check:raise ValueError(f'{name} invalid {d}: {check[:3]}')
 write_csv(out/'comparison.csv',rows)
 write_csv(out/'per_relation.csv',relrows)
 write_csv(out/'errors.csv',errors)
 write_csv(out/'validation.csv',validation,['run','issue'])
 lines=['# NLI Evaluation','',
  'Evaluator: `archive/eval/evaluate.py`; gold: `archive/eval/gold`; canonical texts: `texts/`.',
  f'14 annotated paragraphs; {rows[0]["gold"]} gold facts; overlap spans; paragraph bootstrap {a.bootstrap} replicates (matching the constrained zero-shot evaluation).',
  'NLI sentence→sentence, ensemble, forward threshold **0.90**, reverse **Δ=0.20** frozen before this rerun.',
  '**This is a same-protocol re-evaluation on previously seen development articles, not an independent test.**',
  'NLI supplied facts only: mention/link levels are **not evaluated**, not zero. LELA is an automatically generated entity condition; gold is gold entities. Jev pipeline gold-mention results are from a different pipeline and remain separately reported in `jev/results/scores.md`.','',
  '| Method | Fact P | R | F1 | 95% CI | TP / pred / gold | Fact-pair F1 |',
  '|---|---:|---:|---:|---|---:|---:|']
 for row in rows:
  ci=f'{row["ci_low"]:.0%}–{row["ci_high"]:.0%}' if row['ci_low'] is not None else '—'
  lines.append(f'| {row["run"]} | {row["P"]:.1%} | {row["R"]:.1%} | **{row["F1"]:.1%}** | {ci} | {row["tp"]}/{row["pred"]}/{row["gold"]} | {row["fact_pair_F1"]:.1%} |')
 lines+=['','## Per relation: TP / FP / missed','']
 focus=['nli_gold_reverse_d0p2','nli_lela_reverse_d0p2','zero_shot_sentence_r1','jev_api_gold']
 lines+=['| Relation | '+' | '.join(focus)+' |','|---|'+'---|'*len(focus)]
 d={(x['run'],x['relation']):x for x in relrows}
 for rel in sorted({x['relation'] for x in relrows}):
  cells=[]
  for run in focus:
   x=d.get((run,rel));cells.append(f'{x["tp"]}/{x["fp"]}/{x["missed"]}' if x else '0/0/0')
  lines.append('| '+rel+' | '+' | '.join(cells)+' |')
 lines+=['','## Reproduction','',
   '```bash',
   'python experiments/13_nli_evaluation/evaluation.py',
   'python archive/eval/evaluate.py score --texts texts --gold archive/eval/gold --span overlap --bootstrap 500 --pred experiments/13_nli_evaluation/results/pred/nli_gold_reverse_d0p2 experiments/13_nli_evaluation/results/pred/nli_lela_reverse_d0p2',
   '```','',
   '## Limitations','',
   '- Delta=0.2 and forward threshold were explored on these same articles: no untouched test evidence.',
   '- Only 14 paragraphs; paragraph-bootstrap intervals quantify paragraph sampling, not document-level generalization.',
   '- NLI is evaluated only for facts; gold entities and LELA entities must not be conflated with Jev end-to-end.',
   '- The gold annotations are an unreviewed draft per existing evaluation report.',
   '- `errors.csv` contains the exact matched FP and missed facts for manual investigation.',
   '- Relation naming is matched via the same normalization and entity alignment as the shared evaluator.','']
 (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
 print('\n'.join(lines[:len(rows)+10]))
 print('Saved to',out,'validation issues',len(validation))
 return rows
if __name__=='__main__':main()
