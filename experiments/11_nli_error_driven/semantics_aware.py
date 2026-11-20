#!/usr/bin/env python3
"""Predeclared semantic-policy ablation on existing reverse-probe scores.

No inference, no gold-label-based filtering. Requires project dataset and 09 scores.
All resulting metrics are exploratory (same 2 development articles).
"""
from __future__ import annotations
import argparse, collections, csv, json, sys
from pathlib import Path
from common import ROOT, HERE, DOCS, inputs, measure, predict, read, score_strategy
from reverse_probe import key
from run import SYMMETRIC

POLICIES = {
 'original': frozenset(),
 'contrast_all': frozenset(),
 'exempt_opposed': frozenset({'ex:opposed'}),
 'exempt_opposed_about': frozenset({'ex:opposed','schema:about'}),
 'exempt_about': frozenset({'schema:about'}),
}

def check_pairs(selected, paired):
    keys=[tuple(x['key']) for x in paired]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate reverse-probe keys')
    reverse=dict(zip(keys, paired))
    needed={key(x) for x in selected if x['rel'] not in SYMMETRIC}
    missing=needed-reverse.keys()
    extra=reverse.keys()-needed
    if missing or extra:
        raise ValueError(f'Pair coverage mismatch: missing={len(missing)}, extra={len(extra)}; first_missing={next(iter(missing),None)}')
    for x in selected:
        if x['rel'] in SYMMETRIC:continue
        p=reverse[key(x)]
        if len(p.get('reverse_scores',[]))!=2 or len(p.get('forward_hypothesis_scores',[]))!=2:
            raise ValueError(f'Bad hypothesis scores for {key(x)}')
        if abs(float(p['forward_score'])-float(x['score'])) > 1e-5:
            raise ValueError(f'Forward score mismatch for {key(x)}')
    return reverse

def decisions(selected, reverse, policy, delta, reverse_agg='max'):
    if policy not in POLICIES:raise ValueError(policy)
    kept=[]; audit=[]
    for x in selected:
        rel=x['rel']; base={k:x[k] for k in ('doc','para','subj','obj','subj_entity','obj_entity','rel','score')}
        reason=''
        if policy=='original':reason='original'
        elif rel in SYMMETRIC:reason='symmetric_in_original_pipeline'
        elif rel in POLICIES[policy]:reason='semantics_exemption'
        else:
            p=reverse[key(x)]; rr=p['reverse_scores']
            rs=max(rr) if reverse_agg=='max' else rr[0] if reverse_agg=='baseline' else rr[1]
            diff=x['score']-rs
            reason='contrast_accept' if diff>=delta else 'contrast_reject'
            base.update(reverse_score=rs,margin=diff)
        keep=reason!='contrast_reject'
        base.update(policy=policy,delta=delta,reverse_agg=reverse_agg,kept=int(keep),reason=reason)
        audit.append(base)
        if keep:kept.append(x)
    return kept,audit

def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',choices=('gold','lela'),default=['gold','lela'])
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--delta',type=float,default=.2)
    p.add_argument('--scores',type=Path,default=ROOT/'experiments/09_nli_extended/results')
    p.add_argument('--paired',type=Path,default=HERE/'results/reverse_probe')
    p.add_argument('--out',type=Path,default=HERE/'results/semantics_aware')
    a=p.parse_args(argv)
    schema=read(ROOT/'dataset/relations.json')
    for rel in ['ex:opposed','schema:about']:
        if rel not in schema:raise ValueError(f'Missing schema relation {rel}')
    gold,texts,ev=inputs(); metrics=[]; decisions_all=[]; by_rel=[]; by_doc=[]
    for source in a.sources:
        raw=read(a.scores/f'{source}_sentence_sentence_scores.json')
        chosen=predict(score_strategy(raw,'ensemble'),a.threshold,False)
        stem=f'{source}_ensemble_t{str(a.threshold).replace(".","p")}'
        paired=read(a.paired/f'{stem}_paired_scores.json')
        reverse=check_pairs(chosen,paired)
        for policy in POLICIES:
            kept,audit=decisions(chosen,reverse,policy,a.delta)
            result,preds=measure(raw,kept,source,gold,texts,ev)
            metrics.append({'source':source,'policy':policy,'delta':a.delta,'selected':len(kept),
                'removed_selected':len(chosen)-len(kept),**result})
            decisions_all += audit
            for doc in DOCS:
                one_gold={doc:gold[doc]};one_texts={doc:texts[doc]};one_preds={doc:preds[doc]}
                # reuse common archive evaluator and fact matching on single document
                from compare_all import fact_rows
                d,_=fact_rows(ev,one_texts,one_gold,one_preds,1)
                by_doc.append({'source':source,'policy':policy,'doc':doc,**d})
            counts=collections.Counter((x['rel'],x['reason']) for x in audit)
            for rel in sorted({x['rel'] for x in audit}):
                by_rel.append({'source':source,'policy':policy,'rel':rel,
                    'original_selected':sum(v for (r,reason),v in counts.items() if r==rel),
                    'kept_selected':sum(v for (r,reason),v in counts.items() if r==rel and reason!='contrast_reject'),
                    'rejected_selected':counts[(rel,'contrast_reject')]})
    a.out.mkdir(parents=True,exist_ok=True)
    write_csv(a.out/'comparison.csv',metrics)
    write_csv(a.out/'prediction_audit.csv',decisions_all)
    write_csv(a.out/'by_relation.csv',by_rel)
    write_csv(a.out/'by_document.csv',by_doc)
    lines=['# Semantics-aware counterfactual NLI — exploratory','',
        'Common archive evaluator; 54 gold facts; sentence–sentence ensemble; threshold 0.90.',
        'Gold annotations are used **only for evaluation**, never for filtering.','',
        '| Source | Policy | TP | Pred | FP | Precision | Recall | F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append(f"| {r['source']} | {r['policy']} | {r['tp']} | {r['pred']} | {r['pred']-r['tp']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    lines += ['', '## Scientific cautions','',
      '- `ex:opposed` is semantically mixed: fighting is often mutual, persecution and rebellion are directional. Exemption is an ablation, **not** an asserted ontology truth.',
      '- `schema:about` remains directional according to its schema; exemptions are controls, not semantic corrections.',
      '- Symmetric relations already omitted from the previous reverse probe remain untouched.',
      '- Delta=0.2 and policy choice were explored on the same two papers. These F1 values are **not** held-out estimates.',
      '- This script never resurrects candidate triples removed before original NLI inference.',
      '- Raw selected count can exceed deduplicated prediction count measured by evaluator.','']
    (a.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    for r in metrics:print(f"{r['source']:4} {r['policy']:22} TP={r['tp']:2} pred={r['pred']:3} F1={r['f1']:.4f}")
    print('Wrote',a.out)

if __name__=='__main__':main()
