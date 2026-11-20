#!/usr/bin/env python3
"""Offline, reproducible scoring of NLI score caches and existing Jev/LLM predictions.
Run from repository root: python experiments/09_nli_extended/compare_all.py
No model/API required. Uses archive/eval/evaluate.py as common scoring implementation.
"""
from __future__ import annotations
import argparse, collections, csv, importlib.util, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOCS = ('alford2026', 'nechaeva2017')
SOURCES = ('gold', 'lela')
MODES = ('sentence_sentence', 'sentence_paragraph', 'paragraph_paragraph')
STRATEGIES = ('baseline', 'alternative', 'ensemble')

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def load_json(p):return json.loads(p.read_text(encoding='utf8'))

def safe_write(path, s):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s,encoding='utf8')

def normalize_eids(doc, orig, gold, source):
    """Create canonical entity objects for archive/eval, preserve original IDs in NLI facts.
    Gold uses exact gold EIDs. LELA gets a unique ID per source entity ID.
    """
    if source=='gold':
        return [dict(e, qid=e.get('yago')) for e in gold['entities']]
    e={}
    for j in orig:
        if j['doc']!=doc:continue
        for k in ('subj','obj'):
            eid=j[k+'_entity']; label=j[k]; yago=j.get(k+'_yago')
            if eid not in e:e[eid]={'eid':eid,'label':label,'aliases':[],'qid':yago if isinstance(yago,str) and yago.startswith('yago:') else None}
    return list(e.values())

def convert_nli(doc, orig, selected, gold, source, run):
    return {'doc_id':doc,'entities':normalize_eids(doc,orig,gold,source),'mentions':[],
      'facts':[{'para':j['para'],'subj':j['subj_entity'],'rel':j['rel'],'obj':j['obj_entity']} for j in selected if j['doc']==doc]}

def adapt_existing(obj, gold):
    obj=dict(obj)
    for e in obj.get('entities',[]):
        if e.get('yago') and not e.get('qid'):e['qid']=e['yago']
    return obj

def fact_rows(evaluator, texts, gold, preds, B):
    result=evaluator.evaluate(texts,gold,preds,True,{},B)
    f=result['overall']['fact'];lo,hi=result['ci'].get('fact',(None,None))
    return {'tp':f['tp'],'pred':f['pred'],'gold':f['gold'],'precision':f['P'],'recall':f['R'],'f1':f['F1'],
      'ci_low':lo,'ci_high':hi,'problems':len(result['problems'])},result

def compare(args):
    from run import predict
    ev=load_module('canonical_history_eval',ROOT/'archive/eval/evaluate.py')
    gold={doc:load_json(ROOT/'dataset/gold'/f'{doc}.json') for doc in DOCS}
    texts={doc:{p['id']:p['text'] for p in load_json(ROOT/'dataset/paragraphs'/f'{doc}.json')['paragraphs']} for doc in DOCS}
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];missing=[];per_doc=[];validation=[]
    for source in SOURCES:
      for mode in MODES:
        scores=args.scores/f'{source}_{mode}_scores.json'
        if not scores.exists():
          missing.append({'source':source,'mode':mode,'path':str(scores),'missing_variants':6})
          continue
        jobs=load_json(scores)
        if not isinstance(jobs,list) or not jobs or any(len(j.get('hypothesis_scores',[]))<2 for j in jobs):
          raise ValueError(f'Invalid or incomplete score cache {scores}')
        for strategy in STRATEGIES:
          scored=[]
          for j in jobs:
            k=dict(j);a,b=k['hypothesis_scores'][:2]
            k['score']={'baseline':a,'alternative':b,'ensemble':max(a,b)}[strategy]
            scored.append(k)
          for multi in (False,True):
            chosen=predict(scored,args.threshold,multi)
            run=f'{source}_{mode}_{strategy}_{"multi" if multi else "single"}'
            preds={d:convert_nli(d,jobs,chosen,gold[d],source,run) for d in DOCS}
            for d,p in preds.items():safe_write(out/'pred'/run/f'{d}.json',json.dumps(p,ensure_ascii=False,indent=2)+'\n')
            stats,result=fact_rows(ev,texts,gold,preds,args.bootstrap)
            rows.append({'method':'NLI','run':run,'source':source,'mode':mode,'strategy':strategy,
                 'selection':'multi' if multi else 'single','threshold':args.threshold,**stats,'status':'measured'})
            for d in DOCS:
              s,_=fact_rows(ev,{d:texts[d]},{d:gold[d]},{d:preds[d]},1)
              per_doc.append({'run':run,'doc':d,**s})
            validation.extend({'run':run,'problem':p} for p in result['problems'])
    # Archive predictions: NOT claimed to have identical prompting, entity conditions or thresholds.
    existing={
      'Jev API (archived, gold)':ROOT/'archive/eval/pred/gold__jev-api',
      'Jev API guided (archived, gold)':ROOT/'archive/eval/pred/gold__jev-api-guided',
      'OpenJev (archived, gold)':ROOT/'archive/eval/pred/gold__openjev',
      'Zero-shot LLM sentence r1':ROOT/'archive/experiments/08_constrained_relations/pred/gold__llm-20b-sentence-r1',
      'Zero-shot LLM paragraph r1':ROOT/'archive/experiments/08_constrained_relations/pred/gold__llm-20b-paragraph-r1',
      'Zero-shot LLM paragraph r2':ROOT/'archive/experiments/08_constrained_relations/pred/gold__llm-20b-paragraph-r2',
      'Jev pipeline gold mentions':ROOT/'jev/pred/gold-mentions',
      'Jev pipeline end-to-end':ROOT/'jev/pred',
    }
    for label,path in existing.items():
      if not all((path/f'{d}.json').exists() for d in DOCS):
        missing.append({'source':'reference','mode':label,'path':str(path),'missing_variants':1});continue
      pred={d:adapt_existing(load_json(path/f'{d}.json'),gold[d]) for d in DOCS}
      stats,result=fact_rows(ev,texts,gold,pred,args.bootstrap)
      safe=label.lower().replace(' ','_').replace('(','').replace(')','')
      rows.append({'method':'Jev' if label.startswith(('Jev','OpenJev')) else 'Zero-shot',
       'run':safe,'source':'gold' if 'gold' in label or 'LLM' in label else 'automatic',
       'mode':'archived','strategy':'existing','selection':'provided','threshold':'as recorded',
       **stats,'status':'measured; distinct original pipeline'})
      for d in DOCS:
        s,_=fact_rows(ev,{d:texts[d]},{d:gold[d]},{d:pred[d]},1)
        per_doc.append({'run':safe,'doc':d,**s})
      validation.extend({'run':safe,'problem':p} for p in result['problems'])
    def write_csv(path,items):
      if not items:return
      with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    write_csv(out/'comparison_all.csv',rows)
    write_csv(out/'per_document.csv',per_doc)
    write_csv(out/'missing_scores.csv',missing)
    write_csv(out/'validation_issues.csv',validation)
    best=sorted((r for r in rows if r['method']=='NLI' and r['selection']=='single'),key=lambda x:-x['f1'])[:5]
    nl=[r for r in rows if r['method']=='NLI' and r['source']=='gold']
    report=['# Unified comparison — exploratory benchmark','',
      'All scores below were recomputed from available prediction files / saved NLI entailment scores using `archive/eval/evaluate.py` (`fact`, strict, paragraph-scoped).',
      f'Fixed NLI threshold: **{args.threshold}**. Paragraph-bootstrap 95% intervals: **{args.bootstrap}** replicates.',
      'This is NOT an independent final test: the two papers and 54 manually annotated facts have been repeatedly inspected during development.',
      'Gold annotations explicitly describe themselves as an unreviewed single-annotator draft; validation by a historian is outstanding.',
      '', '## All comparable fact scores','',
      '| Method / run | TP / pred / gold | P | R | F1 | 95% paragraph-bootstrap F1 CI |', '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
      ci='—' if r['ci_low'] is None else f"{r['ci_low']:.1%}–{r['ci_high']:.1%}"
      report.append(f"| {r['run']} | {r['tp']}/{r['pred']}/{r['gold']} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} | {ci} |")
    report+=['','## Missing NLI inference caches','']
    if missing:report+=['| Mode | Missing runs | Expected location |','|---|---:|---|']+[f"| {r['source']} / {r['mode']} | {r['missing_variants']} | `{r['path']}` |" for r in missing]
    else:report.append('None.')
    report+=['','## Research interpretation and comparability','',
      '- **NLI single vs multi:** both are reconstructed from the *same cached inference*, so this is a genuine controlled multi-label ablation. It does not require model reruns.',
      '- **Entity input:** gold and LELA are separate conditions; Jev end-to-end is not the same LELA pipeline. Compare gold with gold, and explicitly distinguish other automatic pipelines.',
      '- **Jev variants:** archived API predictions and the newer Jev pipeline may have different thresholds or implementation details. Their provenance must remain visible.',
      '- **Precision/Recall/F1:** these are exact-match paragraph-level fact scores. The evaluator entity alignment uses unique label/alias and `qid`; predicted `yago` is normalized to `qid` where available.',
      '- **CI:** paragraph bootstrap measures sampling variability, not labeling uncertainty; results from only 14 paragraphs are fragile. Avoid claiming statistical superiority from raw F1.',
      '- **Selection bias:** repeated analysis/tuning on both papers makes every displayed score exploratory. A frozen strategy and untouched independently annotated paper(s) are required for final performance.',
      '- **Reproduction:** `comparison_all.csv` includes the full matrix; `pred/` contains evaluator-format NLI facts for every available cache; missing caches are logged rather than imputed.',
      '', '## Verification', '',f'- Total measured variants: {len(rows)}',f'- Missing cache groups: {len(missing)}',f'- Evaluator validation issue records: {len(validation)}',
      '', '### Next step for final evaluation', '',
      'Freeze the method/model/template/threshold/selection choices **before** annotating and running new, unseen documents. Then rerun the identical evaluator and report the pre-registered test scores separately.']
    safe_write(out/'RESULTS_UNIFIED.md','\n'.join(report)+'\n')
    print(f'Wrote {len(rows)} evaluated variants to {out}; missing groups={len(missing)}; validation issues={len(validation)}')
    for r in best:print(r['run'],f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,default=HERE/'results')
    p.add_argument('--out',type=Path,default=HERE/'results'/'unified')
    p.add_argument('--threshold',type=float,default=0.9)
    p.add_argument('--bootstrap',type=int,default=500)
    compare(p.parse_args())
