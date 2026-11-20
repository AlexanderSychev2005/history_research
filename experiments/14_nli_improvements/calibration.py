"""Experiment 4: score calibration by strict leave-one-DOCUMENT-out evaluation.
No test document labels influence training, feature choice, or threshold.
Two articles = exploratory illustration, not reliable out-of-distribution evidence.
"""
from __future__ import annotations
import argparse,math
from collections import Counter
from pathlib import Path
from shared import *
from type_constraints import categorize,entity_types,passes
from evidence import evidence

def examples(source,raw,base,paired,ev,gold):
    label_map=labels(source,raw,base,ev,gold)
    reverse={tuple(j['key']):max(j['reverse_scores']) for j in paired}
    types=entity_types(source)
    rels=sorted(set(BASE))
    data=[]
    for j in base:
        e=evidence(j);rev=reverse.get(key(j))
        delta=j['score']-rev if rev is not None else 1.0
        s=categorize(types.get((j['doc'],j['subj_entity']),''))
        o=categorize(types.get((j['doc'],j['obj_entity']),''))
        z=[j['score'],min(j['hypothesis_scores']),max(j['hypothesis_scores']),abs(j['hypothesis_scores'][0]-j['hypothesis_scores'][1]),
           (rev if rev is not None else 0.),delta,float(rev is None),float(e['both_names']),
           min((e['distance_chars'] if e['distance_chars'] is not None else 400),400)/400,
           float(e['lexical_cue']),float(passes(j,types))]
        z += [float(j['rel']==r) for r in rels]
        z += [float(s==v) for v in ('person','location','organization','work','unknown')]
        z += [float(o==v) for v in ('person','location','organization','work','unknown')]
        data.append({'key':list(key(j)),'doc':j['doc'],'label':label_map[key(j)],'features':z,'job':j})
    return data

def train_fold(train,test,C):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    import numpy as np
    X=np.asarray([e['features'] for e in train],dtype='float64')
    y=np.asarray([e['label'] for e in train],dtype='int64')
    T=np.asarray([e['features'] for e in test],dtype='float64')
    if not len(train) or not len(test):raise ValueError('Empty train/test split')
    if len(set(y))<2:model=DummyClassifier(strategy='constant',constant=int(y[0]))
    else:model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,class_weight='balanced',random_state=13))
    model.fit(X,y)
    scores=model.predict_proba(T)
    cls=list(model.classes_)
    return [float(s[cls.index(1)]) if 1 in cls else 0. for s in scores]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('gold','lela','both'),default='both')
    p.add_argument('--prob-threshold',type=float,default=.5);p.add_argument('--C',type=float,default=1.)
    p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--out',type=Path,default=HERE/'results'/'calibration')
    a=p.parse_args()
    if not 0<=a.prob_threshold<=1 or a.C<=0:raise ValueError('Invalid frozen hyperparameters')
    ev,gold,texts=gold_and_texts()
    for source in ('gold','lela') if a.source=='both' else (a.source,):
        raw,base,paired=baseline(source);data=examples(source,raw,base,paired,ev,gold)
        rows=[];audit=[];kept=[]
        for held in DOCS:
            tr=[x for x in data if x['doc']!=held];ts=[x for x in data if x['doc']==held]
            if not tr or not ts:raise ValueError('Insufficient examples for cross document evaluation')
            probs=train_fold(tr,ts,a.C)
            selected=[x['job'] for x,prob in zip(ts,probs) if prob>=a.prob_threshold]
            kept.extend(selected)
            baseline_doc=[x['job'] for x in ts]
            for policy,jobs in [('frozen_baseline',baseline_doc),('cross_document_calibrated',selected)]:
                rows.append({'source':source,'test_doc':held,'train_docs':','.join(d for d in DOCS if d!=held),'policy':policy,
                    **evaluate(source,raw,jobs,ev,gold,texts,a.bootstrap,[held])})
            for x,prob in zip(ts,probs):audit.append({'source':source,'doc':held,'train_docs':','.join(d for d in DOCS if d!=held),
                'key':json.dumps(x['key'],ensure_ascii=False),'gold_label_for_audit_only':x['label'],'predicted_probability':prob,
                'accepted':prob>=a.prob_threshold})
        # Pooled out-of-fold metric (each document was unseen by its corresponding fitted model).
        rows.append({'source':source,'test_doc':'POOLED_OUT_OF_FOLD','train_docs':'leave-one-document-out','policy':'frozen_baseline',
             **evaluate(source,raw,base,ev,gold,texts,a.bootstrap)})
        rows.append({'source':source,'test_doc':'POOLED_OUT_OF_FOLD','train_docs':'leave-one-document-out','policy':'cross_document_calibrated',
             **evaluate(source,raw,kept,ev,gold,texts,a.bootstrap)})
        table(a.out/f'{source}_comparison.csv',rows);table(a.out/f'{source}_prediction_audit.csv',audit)
        for r in rows:print(source,r['test_doc'],r['policy'],f'F1={r["f1"]:.4f}',f'TP={r["tp"]}, FP={r["fp"]}')
if __name__=='__main__':main()
