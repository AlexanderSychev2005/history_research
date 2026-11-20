#!/usr/bin/env python3
"""New inference ablation: original vs entity-marked sentence NLI.

Requires original repo inputs, Torch and Transformers, and generates scores
for BOTH premise variants without changing candidate generation or templates.
Run: python experiments/10_nli_precision/focused_inference.py --source gold --batch-size 8
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/09_nli_extended'))
from run import DOCS, MODEL, candidate_jobs, infer, read

def mark_entities(text,subject,object_):
    """Mark a disjoint, exact surface occurrence of each entity.
    Skip ambiguous/overlapping cases; avoids changing candidate universe silently.
    """
    def spans(label):
        if not label:return []
        return [(m.start(),m.end()) for m in re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)',text,re.I)]
    ss=spans(subject);oo=spans(object_)
    if not ss or not oo:return None
    pairs=[(a,b) for a in ss for b in oo if a[1]<=b[0] or b[1]<=a[0]]
    if not pairs:return None
    # pick closest non-overlapping pair deterministically
    a,b=min(pairs,key=lambda pair:(max(pair[0][0],pair[1][0])-min(pair[0][1],pair[1][1]) if (pair[0][1]<=pair[1][0] or pair[1][1]<=pair[0][0]) else 0,pair[0][0],pair[1][0]))
    chunks=[(a,'[SUBJ] ',' [/SUBJ]'),(b,'[OBJ] ',' [/OBJ]')]
    out=text
    for (start,end),left,right in sorted(chunks,key=lambda x:x[0][0],reverse=True):
        out=out[:start]+left+out[start:end]+right+out[end:]
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',choices=('gold','lela'),default='gold')
    ap.add_argument('--model',default=MODEL)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    ap.add_argument('--max-length',type=int,default=512)
    ap.add_argument('--out',type=Path,default=HERE/'results'/'focused')
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    schema=read(ROOT/'dataset/relations.json')
    raw=[j for doc in DOCS for j in candidate_jobs(doc,a.source,'sentence',False,schema)]
    # Match the original candidate set: two variants share exact same job identifiers.
    marked=[];skipped=0
    for j in raw:
        m=mark_entities(j['text'],j['subj'],j['obj'])
        if m is None:skipped+=1;continue
        k=dict(j);k['text']=m;marked.append(k)
    print(f'Prepared {len(raw)} original NLI jobs and {len(marked)} markable jobs; unmarkable={skipped}')
    if not marked:raise ValueError('No markable entity pairs; examine mention alignment')
    # Fair comparison on SAME markable subset only.
    keys={(j['doc'],j['para'],j['start'],j['end'],j['subj_entity'],j['obj_entity'],j['rel']) for j in marked}
    original=[dict(j) for j in raw if (j['doc'],j['para'],j['start'],j['end'],j['subj_entity'],j['obj_entity'],j['rel']) in keys]
    assert len(original)==len(marked)
    if not a.prepare_only:
        infer(a.model,original,'ensemble',a.batch_size,a.device,a.max_length)
        infer(a.model,marked,'ensemble',a.batch_size,a.device,a.max_length)
    for label,data in (('original_matched',original),('entity_marked',marked)):
        (a.out/f'{a.source}_{label}_{"candidates" if a.prepare_only else "scores"}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Saved paired matched outputs in',a.out)
    if not a.prepare_only:
        print('To compare them with the exact evaluator, use compare_focused.py; keep this new experiment separate from old 09 caches.')
if __name__=='__main__':main()
