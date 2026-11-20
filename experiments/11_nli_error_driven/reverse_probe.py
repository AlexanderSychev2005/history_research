#!/usr/bin/env python3
"""NEW inference: test whether sentence NLI is sensitive to reversing subject/object roles.
For each predicted relation, score a counterfactual hypothesis with entity roles swapped
against the SAME original premise. Inputs are existing gold or LELA cached NLI scores.
Run --prepare-only before inference. Score only the baseline or ensemble selected jobs.
"""
from __future__ import annotations
import argparse,collections,json,sys
from pathlib import Path
from common import ROOT,HERE,read,score_strategy,predict
from run import BASE,ALT,SYMMETRIC,MODEL,infer

def selected_for_probe(raw,strategy,threshold):
    scored=score_strategy(raw,strategy)
    chosen=predict(scored,threshold,False)
    # A semantic contrast is ill-defined for symmetric relations.
    return [x for x in chosen if x['rel'] not in SYMMETRIC and x['subj'] != x['obj']]

def reversed_jobs(chosen):
    """Swap the template roles, NOT the original evidence. Original metadata remains intact."""
    jobs=[]
    for x in chosen:
        z=dict(x);z['subj'],z['obj']=x['obj'],x['subj']
        z.pop('score',None);z.pop('hypothesis_scores',None)
        jobs.append(z)
    return jobs

def key(j):return (j['doc'],j['para'],j['subj_entity'],j['rel'],j['obj_entity'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('gold','lela'),default='gold')
    p.add_argument('--strategy',choices=('baseline','alternative','ensemble'),default='ensemble')
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--scores',type=Path,default=ROOT/'experiments/09_nli_extended/results')
    p.add_argument('--out',type=Path,default=HERE/'results'/'reverse_probe')
    p.add_argument('--model',default=MODEL);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    p.add_argument('--max-length',type=int,default=512);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=read(a.scores/f'{a.source}_sentence_sentence_scores.json')
    chosen=selected_for_probe(raw,a.strategy,a.threshold)
    jobs=reversed_jobs(chosen)
    print(f'Selected {len(chosen)} asymmetric predictions for paired role-reversal probes')
    manifest=[]
    for x,z in zip(chosen,jobs):
        manifest.append({'key':list(key(x)),'original_subject':x['subj'],'original_object':x['obj'],
          'forward_score':x['score'],'forward_hypothesis_scores':x['hypothesis_scores'],
          'forward_base_hypothesis':BASE[x['rel']].format(s=x['subj'],o=x['obj']),
          'reverse_base_hypothesis':BASE[x['rel']].format(s=z['subj'],o=z['obj']),
          'forward_alt_hypothesis':ALT[x['rel']].format(s=x['subj'],o=x['obj']),
          'reverse_alt_hypothesis':ALT[x['rel']].format(s=z['subj'],o=z['obj'])})
    stem=f'{a.source}_{a.strategy}_t{str(a.threshold).replace(".","p")}'
    (a.out/f'{stem}_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    if a.prepare_only:
        print('Prepared manifests. No inference performed.');return
    if not jobs:raise RuntimeError('No eligible asymmetric predictions')
    # run.infer always uses BASE/ALT to generate hypotheses and sees the swapped surfaces.
    infer(a.model,jobs,'ensemble',a.batch_size,a.device,a.max_length)
    reverse={key(x):{'reverse_scores':x['hypothesis_scores']} for x in jobs}
    assert len(reverse)==len(jobs),'Duplicate keys; inspect pair deduplication'
    output=[]
    for row in manifest:
        k=tuple(row['key']);row.update(reverse[k]);output.append(row)
    path=a.out/f'{stem}_paired_scores.json'
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Saved',path)
if __name__=='__main__':main()
