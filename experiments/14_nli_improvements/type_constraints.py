"""Experiment 1: high-precision entity type compatibility on frozen predictions.
Unknown types are never rejected; no gold labels used for decisions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from shared import *
# Conservative rules; avoid assuming every historical grouping is a modern organization.
# Using labels rather than specific proper names makes the rule portable.
ROLE_TYPES={
 'schema:birthPlace':('person','location'), 'schema:deathPlace':('person','location'),
 'yago:burialPlace':('person','location'), 'ex:captiveIn':('person','location'),
 'yago:hasMother':('person','person'), 'yago:hasFather':('person','person'),
 'schema:spouse':('person','person'), 'yago:studentOf':('person','person'),
 'schema:worksFor':('person','person_or_organization'),
 'yago:creator':('work','person'), # Schema here has reversed creator template: object is creator.
 'schema:about':('work','any'),
}

def categorize(t):
    t=str(t).strip().lower().replace('_',' ')
    if not t:return 'unknown'
    if any(w in t for w in ['creative work','written work','book','text','literary work','document','publication']):return 'work'
    if any(w in t for w in ['person','human','historian','author','ruler','king']):return 'person'
    if any(w in t for w in ['place','location','geograph','city','region','country','empire','province','river']):return 'location'
    if any(w in t for w in ['organization','organisation','institution','church','army','group','dynasty']):return 'organization'
    return 'unknown'

def permits(actual,expected):
    if actual=='unknown' or expected=='any':return True
    return actual==expected or (expected=='person_or_organization' and actual in ('person','organization'))

def passes(j,types):
    expected=ROLE_TYPES.get(j['rel'])
    if expected is None:return True
    subj=categorize(types.get((j['doc'],j['subj_entity']),''))
    obj=categorize(types.get((j['doc'],j['obj_entity']),''))
    return permits(subj,expected[0]) and permits(obj,expected[1])

def entity_types(source):
    out={}
    for d in DOCS:
        g=read(ROOT/'dataset/gold'/f'{d}.json')
        if source=='gold':
            out.update({(d,e['eid']):e['type'] for e in g['entities']})
        else:
            for m in mentions(d,source,g):
                out[(d,m['entity'])]=m.get('type','')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('gold','lela','both'),default='both')
    p.add_argument('--out',type=Path,default=HERE/'results'/'types');p.add_argument('--bootstrap',type=int,default=100)
    args=p.parse_args()
    for source in DOC_SOURCES(args.source):
        raw,base,_=baseline(source);types=entity_types(source)
        selected=[j for j in base if passes(j,types)]
        rows=report(args.out,'type_constraints',source,raw,base,{'frozen_baseline':base,'conservative_types':selected},args.bootstrap)
        table(args.out/f'{source}_rejected.csv',[{'doc':j['doc'],'para':j['para'],'subj':j['subj'],'rel':j['rel'],'obj':j['obj'],'subject_type':types.get((j['doc'],j['subj_entity']),''),'object_type':types.get((j['doc'],j['obj_entity']),'')} for j in base if not passes(j,types)],['doc','para','subj','rel','obj','subject_type','object_type'])
        for r in rows:print(source,r['policy'],f'F1={r["f1"]:.4f}',f'TP={r["tp"]}, FP={r["fp"]}')

def DOC_SOURCES(v):return ('gold','lela') if v=='both' else (v,)
if __name__=='__main__':main()
