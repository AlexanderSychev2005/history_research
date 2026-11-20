"""Experiment 3: evidence audit and interpretable, zero-inference evidence filters.
Evidence flags are heuristic proxies, not claims of definitive textual entailment.
Retain evidence snippet for every prediction, even the ones rejected by a rule.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
from shared import *
# Lexical triggers are deliberately weak features; report them as heuristic evidence only.
CUES={
 'schema:worksFor':r'\b(served|service|employed|worked|commanded|under|subordinate|officer)\b',
 'yago:creator':r'\b(wrote|written|author|authored|composed|created|work|history)\b',
 'schema:location':r'\b(in|at|based|from|located|lived|resided|settled|city|region)\b',
 'schema:about':r'\b(about|concerning|account|describ|history|discuss|focus|subject)\b',
 'yago:partOf':r'\b(part|within|belong|member|included|composed|consisted)\b',
 'ex:captiveIn':r'\b(captive|prison|exile|deport|held|detain)\b',
 'ex:opposed':r'\b(fought|against|oppos|rebel|battle|war|persecut)\b',
}

def evidence(j):
    txt=j['text'];s=j['subj'];o=j['obj']
    def occurrences(word):return [(m.start(),m.end()) for m in re.finditer(re.escape(word),txt,re.IGNORECASE)] if word else []
    aa=occurrences(s);bb=occurrences(o)
    pairs=[(max(0,max(a[0],b[0])-min(a[1],b[1])),a,b) for a in aa for b in bb]
    if pairs:
        dist,a,b=min(pairs,key=lambda q:q[0]);l=max(0,min(a[0],b[0])-90);r=min(len(txt),max(a[1],b[1])+90)
        snippet=txt[l:r].strip()
    else:dist=None;snippet=txt[:280].strip()
    trigger=CUES.get(j['rel'])
    has_cue=bool(trigger and re.search(trigger,snippet,re.IGNORECASE))
    return {'both_names':bool(pairs),'distance_chars':dist,'lexical_cue':has_cue,'snippet':snippet}

def policy(j,ev,mode,max_distance):
    if mode=='both_mentions':return ev['both_names']
    if mode=='nearby':return ev['both_names'] and ev['distance_chars']<=max_distance
    if mode=='nearby_with_cue':
        return ev['both_names'] and ev['distance_chars']<=max_distance and (j['rel'] not in CUES or ev['lexical_cue'])
    raise ValueError(mode)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('gold','lela','both'),default='both')
    p.add_argument('--distance',type=int,default=120);p.add_argument('--bootstrap',type=int,default=100)
    p.add_argument('--out',type=Path,default=HERE/'results'/'evidence');a=p.parse_args()
    if a.distance<0:raise ValueError('Distance must be non-negative')
    for source in ('gold','lela') if a.source=='both' else (a.source,):
        raw,base,_=baseline(source)
        assessed={key(j):evidence(j) for j in base}
        variants={'frozen_baseline':base}
        for mode in ('both_mentions','nearby','nearby_with_cue'):
            variants[mode]=[j for j in base if policy(j,assessed[key(j)],mode,a.distance)]
        rows=report(a.out,'evidence',source,raw,base,variants,a.bootstrap)
        audits=[{'doc':j['doc'],'para':j['para'],'subj':j['subj'],'rel':j['rel'],'obj':j['obj'],
                  **assessed[key(j)],**{f'kept_{mode}':policy(j,assessed[key(j)],mode,a.distance) for mode in ('both_mentions','nearby','nearby_with_cue')}} for j in base]
        table(a.out/f'{source}_evidence_audit.csv',audits)
        for row in rows:print(source,row['policy'],f'F1={row["f1"]:.4f}',f'TP={row["tp"]}, FP={row["fp"]}')
if __name__=='__main__':main()
