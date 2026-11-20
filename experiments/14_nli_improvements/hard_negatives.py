"""Experiment 2: relation-specific hard-negative hypotheses on frozen candidate facts.
--prepare-only generates an inference manifest without loading a model.
--infer runs MNLI and caches probabilities; --evaluate uses existing cache.
The hypothesis is a plausible competing interpretation, NOT necessarily a logical negation.
"""
from __future__ import annotations
import argparse,copy
from pathlib import Path
from shared import *
# Only relations with interpretable alternatives; all other relations remain as frozen baseline.
HARD_NEGATIVE={
 'schema:about':'{s} merely mentions {o} without being primarily about {o}.',
 'yago:partOf':'{s} is associated with {o} but is not a constituent part of {o}.',
 'schema:worksFor':'{s} met or opposed {o}, but did not serve {o}.',
 'schema:location':'{s} travelled through {o} without being based in {o}.',
 'yago:creator':'{o} was mentioned in {s} but did not author {s}.',
}

def jobs_for(base):
    return [{'key':list(key(j)),'doc':j['doc'],'para':j['para'],'premise':j['text'],
             'positive':BASE[j['rel']].format(s=j['subj'],o=j['obj']),
             'hard_negative':HARD_NEGATIVE[j['rel']].format(s=j['subj'],o=j['obj']),
             'relation':j['rel']} for j in base if j['rel'] in HARD_NEGATIVE]

def infer_requests(jobs,model,device,batch_size,max_length):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    import torch
    tok=AutoTokenizer.from_pretrained(model)
    net=AutoModelForSequenceClassification.from_pretrained(model)
    labels={int(i):str(v).lower() for i,v in net.config.id2label.items()}
    entail=next((i for i,v in labels.items() if v=='entailment'),None)
    if entail is None:raise ValueError(f'No entailment label: {labels}')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    net=net.to(device).eval();result=[]
    for start in range(0,len(jobs),batch_size):
        batch=jobs[start:start+batch_size]
        token=tok([j['premise'] for j in batch],[j['hard_negative'] for j in batch],padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(device)
        with torch.inference_mode():scores=torch.softmax(net(**token).logits,dim=-1)[:,entail].cpu().tolist()
        for j,score in zip(batch,scores):result.append({**j,'negative_score':float(score)})
        if start==0 or (start//batch_size)%50==0:print('scored',min(start+batch_size,len(jobs)),'/',len(jobs),flush=True)
    return result

def filter_negative(base,scores,max_negative=None,margin=None):
    lookup={tuple(j['key']):j['negative_score'] for j in scores}
    accepted=[]
    for j in base:
        if j['rel'] not in HARD_NEGATIVE:accepted.append(j);continue
        k=key(j)
        if k not in lookup:raise KeyError(f'Missing hard-negative score for {k}')
        ns=lookup[k]
        if (max_negative is None or ns <= max_negative) and (margin is None or j['score']-ns >= margin):accepted.append(j)
    return accepted

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('gold','lela'),default='gold')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--infer',action='store_true')
    p.add_argument('--model',default=MODEL);p.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--max-negative',type=float,default=.5);p.add_argument('--margin',type=float,default=.2)
    p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--out',type=Path,default=HERE/'results'/'hard_negatives')
    a=p.parse_args();raw,base,_=baseline(a.source);jobs=jobs_for(base)
    stem=a.source;write(a.out/f'{stem}_manifest.json',jobs)
    if a.prepare_only:
        print('Prepared',len(jobs),'hard-negative jobs in',a.out);return
    scored_path=a.out/f'{stem}_scores.json'
    if a.infer:
        write(scored_path,infer_requests(jobs,a.model,a.device,a.batch_size,a.max_length))
    scores=read(scored_path)
    if {tuple(j['key']) for j in scores}!={tuple(j['key']) for j in jobs}:raise ValueError('Cached scores do not match current manifest')
    variants={'frozen_baseline':base,
     'max_negative':filter_negative(base,scores,max_negative=a.max_negative),
     'forward_minus_negative':filter_negative(base,scores,margin=a.margin),
     'both_rules':filter_negative(base,scores,max_negative=a.max_negative,margin=a.margin)}
    rows=report(a.out,'hard_negative',a.source,raw,base,variants,a.bootstrap)
    for r in rows:print(r['policy'],f'F1={r["f1"]:.4f}',f'TP={r["tp"]}, FP={r["fp"]}')
if __name__=='__main__':main()
