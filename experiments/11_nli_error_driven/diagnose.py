#!/usr/bin/env python3
"""Diagnostic error taxonomy from saved 09 NLI scores, not a new test score.
Usage: python experiments/11_nli_error_driven/diagnose.py
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
from common import ROOT,HERE,DOCS,read,save,inputs,score_strategy,measure,normalized_keys,predict

def classify(k,gold_keys):
    para,subj,rel,obj=k
    if subj.startswith('?') or obj.startswith('?'):return 'unmatched_entity'
    if (para,obj,rel,subj) in gold_keys:return 'reversed_roles_same_relation'
    if any(x[0]==para and x[1]==subj and x[3]==obj for x in gold_keys):return 'wrong_relation_same_directed_pair'
    if any(x[0]==para and x[1]==obj and x[3]==subj for x in gold_keys):return 'wrong_relation_reversed_pair'
    return 'no_gold_fact_for_pair_in_paragraph'

def diagnose(raw,source,strategy,threshold,gold,texts,ev):
    scored=score_strategy(raw,strategy);chosen=predict(scored,threshold,False)
    result,pred=measure(raw,chosen,source,gold,texts,ev)
    rows=[];relation=collections.defaultdict(lambda:collections.Counter());counts=collections.Counter();fns=[]
    for d in DOCS:
        g,p=normalized_keys(ev,gold,pred,d)
        for k in p:
            status='TP' if k in g else 'FP';category='' if status=='TP' else classify(k,g)
            counts[status]+=1
            if category:counts[category]+=1
            relation[k[2]][status]+=1
            f=p[k]
            related=[x for x in scored if x['doc']==d and x['para']==k[0] and x['rel']==f['rel'] and x['subj_entity']==f['subj'] and x['obj_entity']==f['obj']]
            best=max((x['score'] for x in related),default=None)
            rows.append({'doc':d,'para':k[0],'subj_id':k[1],'rel':f['rel'],'obj_id':k[3],
              'status':status,'error_category':category,'score':best,'subj_mention':max((x['subj'] for x in related),default=''),
              'obj_mention':max((x['obj'] for x in related),default=''),
              'evidence_sentence':max((x['text'] for x in related),default='')})
        for k in g.keys()-p.keys():
            # Coverage: exact gold triple present in scored candidate set after entity alignment is
            # computed separately; this list identifies false negatives, not their cause.
            fns.append({'doc':d,'para':k[0],'subj_id':k[1],'rel':k[2],'obj_id':k[3]})
    relrows=[{'rel':r,'tp':c['TP'],'fp':c['FP'],'precision':round(c['TP']/(c['TP']+c['FP']),4) if c['TP']+c['FP'] else 0} for r,c in sorted(relation.items(),key=lambda x:-x[1]['FP'])]
    return result,rows,relrows,fns,dict(counts)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores',type=Path,default=ROOT/'experiments/09_nli_extended/results')
    ap.add_argument('--out',type=Path,default=HERE/'results'/'diagnostics')
    ap.add_argument('--source',choices=('gold','lela'),default='gold')
    ap.add_argument('--mode',choices=('sentence_sentence','sentence_paragraph','paragraph_paragraph'),default='sentence_sentence')
    ap.add_argument('--strategy',choices=('baseline','alternative','ensemble'),default='ensemble')
    ap.add_argument('--threshold',type=float,default=.9)
    a=ap.parse_args();path=a.scores/f'{a.source}_{a.mode}_scores.json'
    if not path.exists():raise FileNotFoundError(path)
    gold,texts,ev=inputs();raw=read(path)
    result,rows,relrows,fns,counts=diagnose(raw,a.source,a.strategy,a.threshold,gold,texts,ev)
    a.out.mkdir(parents=True,exist_ok=True)
    save(a.out/'predictions_with_errors.csv',rows)
    save(a.out/'relation_errors.csv',relrows)
    save(a.out/'false_negatives.csv',fns)
    (a.out/'summary.json').write_text(json.dumps({'condition':vars(a)|{'scores':str(a.scores),'out':str(a.out)},'metrics':result,'error_categories':counts},indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
    lines=['# NLI error-driven diagnostics','',f"Condition: {a.source}/{a.mode}/{a.strategy}; threshold={a.threshold}",
           f"P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}; TP={result['tp']} FP={result['pred']-result['tp']} FN={result['gold']-result['tp']}",
           '', '## False-positive categories (descriptive, not causal explanations)']
    lines += [f'- {k}: {v}' for k,v in counts.items() if k not in ('TP','FP')]
    lines+=['','## Most frequent false-positive relations','', '| Relation | TP | FP | Precision |','|---|---:|---:|---:|']
    lines += [f"| {r['rel']} | {r['tp']} | {r['fp']} | {r['precision']:.1%} |" for r in relrows[:15]]
    lines+=['','Categories rely on the *existing gold annotations*: `no_gold_fact_for_pair_in_paragraph` does not prove the prediction is factually false. Inspect manually before interpreting causally.',
      'Do NOT use gold-derived error categories as a prediction-time filter. The same two articles were already used for development.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f"Diagnostics {a.source}/{a.mode}/{a.strategy}: TP={result['tp']} FP={result['pred']-result['tp']} FN={result['gold']-result['tp']} F1={result['f1']:.3f}")
    print('Top FP categories:',counts);print('Saved:',a.out)
if __name__=='__main__': main()
