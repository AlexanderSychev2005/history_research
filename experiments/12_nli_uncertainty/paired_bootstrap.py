#!/usr/bin/env python3
"""Paired, paragraph-cluster bootstrap for NLI reverse-probe ablations.

Runs without model inference. Resamples the same paragraph IDs for the two
methods in each replicate. Reports descriptive uncertainty, NOT a held-out test.
Uses the existing archive evaluator for fact scoring and entity alignment.
"""
from __future__ import annotations
import argparse, collections, csv, json, math, random, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/11_nli_error_driven'))
from common import DOCS, inputs, measure, read, score_strategy, normalized_keys
from semantics_aware import check_pairs, decisions
from reverse_probe import key
from run import predict


def f1(t):
    tp,pred,gold=t
    return 2*tp/(pred+gold) if pred+gold else 1.0


def quantile(v,p):
    a=sorted(v); position=(len(a)-1)*p; lo=int(position);hi=min(len(a)-1,lo+1)
    return a[lo]+(a[hi]-a[lo])*(position-lo)


def paired_bootstrap(units, replicates=5000, seed=13, stratified=True):
    if replicates<100:raise ValueError('At least 100 replicates required to estimate percentile intervals')
    if not units:raise ValueError('No evaluated paragraphs')
    groups=collections.defaultdict(list)
    for u in units:groups[u['doc']].append(u)
    groups=list(groups.values()) if stratified else [units]
    rng=random.Random(seed)
    base=tuple(sum(u['original'][k] for u in units) for k in range(3))
    rev=tuple(sum(u['contrast'][k] for u in units) for k in range(3))
    delta=f1(rev)-f1(base)
    samples=[]; originals=[]; contrasts=[]
    for _ in range(replicates):
        ss=[rng.choice(group) for group in groups for _ in group]
        a=tuple(sum(u['original'][k] for u in ss) for k in range(3))
        b=tuple(sum(u['contrast'][k] for u in ss) for k in range(3))
        fa,fb=f1(a),f1(b)
        originals.append(fa);contrasts.append(fb);samples.append(fb-fa)
    return {'n_paragraphs':len(units),'n_documents':len(groups) if stratified else len(set(u['doc'] for u in units)),
            'replicates':replicates,'seed':seed,'stratified_by_document':stratified,
            'original_f1':f1(base),'contrast_f1':f1(rev),'delta_f1':delta,
            'original_ci95':[quantile(originals,.025),quantile(originals,.975)],
            'contrast_ci95':[quantile(contrasts,.025),quantile(contrasts,.975)],
            'delta_ci95':[quantile(samples,.025),quantile(samples,.975)],
            'fraction_bootstrap_delta_positive':sum(x>0 for x in samples)/replicates,
            'original_counts':{'tp':base[0],'pred':base[1],'gold':base[2]},
            'contrast_counts':{'tp':rev[0],'pred':rev[1],'gold':rev[2]}}


def paragraph_units(ev, gold, texts, pred_a, pred_b):
    result=[]
    for d in DOCS:
        aa=ev.align(gold[d]['entities'],pred_a[d]['entities'])
        bb=ev.align(gold[d]['entities'],pred_b[d]['entities'])
        for p in gold[d]['scope']['paragraphs']:
            ca=ev.count(gold[d],pred_a[d],aa,p,texts[d][p],True,{})
            cb=ev.count(gold[d],pred_b[d],bb,p,texts[d][p],True,{})
            a=(ca['fact_tp'],ca['fact_pred'],ca['fact_gold'])
            b=(cb['fact_tp'],cb['fact_pred'],cb['fact_gold'])
            if a[2]!=b[2]:raise AssertionError('Gold changed between paired runs')
            result.append({'doc':d,'para':p,'original':a,'contrast':b})
    return result


def audit_one(ev, gold, pred_a, pred_b, source):
    rows=[]
    for d in DOCS:
        g,a=normalized_keys(ev,gold,pred_a,d)
        _,b=normalized_keys(ev,gold,pred_b,d)
        orig=set(a);contrast=set(b)
        if not contrast<=orig:raise AssertionError('Contrast probe unexpectedly added predictions')
        for k in sorted(orig, key=str):
            f=a[k]
            rows.append({'source':source,'doc':d,'para':f['para'],
                  'subj_entity':f['subj'],'rel':f['rel'],'obj_entity':f['obj'],
                  'was_tp':int(k in g),'kept':int(k in contrast),
                  'effect':('retained_TP' if k in g else 'retained_FP') if k in contrast else ('lost_TP' if k in g else 'removed_FP')})
    return rows


def save_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    gold,texts,ev=inputs()
    allstats={}; auditor=[]; rel_rows=[];para_rows=[]
    for source in args.sources:
        raw=read(args.scores/f'{source}_sentence_sentence_scores.json')
        chosen=predict(score_strategy(raw,'ensemble'),args.threshold,False)
        stem=f'{source}_ensemble_t{str(args.threshold).replace(".","p")}'
        paired=read(args.paired/f'{stem}_paired_scores.json')
        rev=check_pairs(chosen,paired)
        filtered,_=decisions(chosen,rev,'contrast_all',args.delta)
        orig_metrics,orig_pred=measure(raw,chosen,source,gold,texts,ev)
        contrast_metrics,contrast_pred=measure(raw,filtered,source,gold,texts,ev)
        paras=paragraph_units(ev,gold,texts,orig_pred,contrast_pred)
        stat=paired_bootstrap(paras,args.bootstrap,args.seed,args.stratified)
        assert (stat['original_counts']['tp'],stat['original_counts']['pred'],stat['original_counts']['gold']) == (orig_metrics['tp'],orig_metrics['pred'],orig_metrics['gold'])
        assert (stat['contrast_counts']['tp'],stat['contrast_counts']['pred'],stat['contrast_counts']['gold']) == (contrast_metrics['tp'],contrast_metrics['pred'],contrast_metrics['gold'])
        allstats[source]=stat
        for u in paras:
            para_rows.append({'source':source,'doc':u['doc'],'para':u['para'],
                        'original_tp':u['original'][0],'original_pred':u['original'][1],
                        'contrast_tp':u['contrast'][0],'contrast_pred':u['contrast'][1],
                        'gold':u['original'][2]})
        audit=audit_one(ev,gold,orig_pred,contrast_pred,source);auditor+=audit
        if len(audit)!=orig_metrics['pred']:raise AssertionError('Audit and evaluator prediction counts diverge')
        effect=collections.Counter(x['effect'] for x in audit)
        if effect['lost_TP']!=orig_metrics['tp']-contrast_metrics['tp'] or effect['removed_FP']!=orig_metrics['pred']-contrast_metrics['pred']-effect['lost_TP']:
            raise AssertionError('Prediction delta does not match audit')
        for relation in sorted(set(x['rel'] for x in audit)):
            chunk=[x for x in audit if x['rel']==relation];cnt=collections.Counter(x['effect'] for x in chunk)
            rel_rows.append({'source':source,'rel':relation,'original_pred':len(chunk),
                   'removed_fp':cnt['removed_FP'],'lost_tp':cnt['lost_TP'],
                   'kept_fp':cnt['retained_FP'],'kept_tp':cnt['retained_TP']})
        print(f"{source}: F1 {stat['original_f1']:.4f} -> {stat['contrast_f1']:.4f}; delta={stat['delta_f1']:+.4f}; bootstrap 95% CI [{stat['delta_ci95'][0]:+.4f},{stat['delta_ci95'][1]:+.4f}]")
    args.out.mkdir(parents=True,exist_ok=True)
    save_csv(args.out/'per_paragraph.csv',para_rows)
    save_csv(args.out/'prediction_audit.csv',auditor)
    save_csv(args.out/'by_relation.csv',rel_rows)
    (args.out/'bootstrap.json').write_text(json.dumps(allstats,indent=2,ensure_ascii=False)+'\n')
    lines=['# Paired paragraph-bootstrap analysis (exploratory)','',
         'The same paragraph is sampled into BOTH methods in every replicate; documents are stratified by default.',
         'These are percentile intervals for F1 and the paired F1 difference (contrast − original), not a new independent-test estimate.',
         'All fact counts and entity alignment use the archive evaluator, with the same gold 54 facts.',
         '', '| Source | Original F1 (95% bootstrap CI) | Contrast F1 (95% bootstrap CI) | Paired F1 gain (95% CI) |',
         '|---|---:|---:|---:|']
    for source,st in allstats.items():
        ci=lambda k: f"{st[k][0]:.1%}–{st[k][1]:.1%}"
        lines.append(f"| {source} | {st['original_f1']:.1%} ({ci('original_ci95')}) | {st['contrast_f1']:.1%} ({ci('contrast_ci95')}) | {st['delta_f1']:+.1%} ({st['delta_ci95'][0]:+.1%} to {st['delta_ci95'][1]:+.1%}) |")
    lines+=['','## Interpretation','',
       '- **Correct CI implementation:** previous files computed `ci_low=ci_high` because the evaluator was invoked with **one bootstrap replicate** (`B=1`) for speed. Those numbers are not confidence intervals. This patch uses at least 100 replicates (default 5000).',
       '- Bootstrap paragraphs (cluster level), not individual triples; the two methods use exactly the same resampling indices.',
       '- If the CI for F1 gain includes zero, the measured improvement is not clearly separated from zero under this resampling model.',
       '- Even if the interval excludes zero, delta=0.2 was selected after exploring these same papers; the interval does not remove selection bias.',
       '- `prediction_audit.csv` separates deduplicated removed FP versus lost TP. A gold-absent fact is a benchmark FP, not proof of historical falsity.',
       '- `ex:opposed` is semantically mixed; these are diagnostic results, not evidence that every opposition relation is directional.',
       '- The archive gold set is small and was previously inspected; independent document-level testing is still required.','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print('Saved',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',choices=('gold','lela'),default=['gold','lela'])
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--delta',type=float,default=.2)
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--stratified',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--scores',type=Path,default=ROOT/'experiments/09_nli_extended/results')
    p.add_argument('--paired',type=Path,default=ROOT/'experiments/11_nli_error_driven/results/reverse_probe')
    p.add_argument('--out',type=Path,default=HERE/'results')
    run(p.parse_args())

if __name__=='__main__':main()
