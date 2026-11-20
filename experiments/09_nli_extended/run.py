"""Fresh NLI inference: hypothesis verbalization, context, and LELA vs gold.
Run from project root: python experiments/09_nli_extended/run.py --help
"""
from __future__ import annotations
import argparse, collections, csv, itertools, json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOCS = ('alford2026', 'nechaeva2017')
MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
# Only entity-object relations. Templates are relation-specific, no date literals.
BASE = {
'yago:creator':'{o} created {s}.', 'schema:location':'{s} is located in {o}.',
'schema:about':'{s} is about {o}.','yago:beliefSystem':'{s} follows {o} as a religion or belief.',
'yago:influencedBy':'{s} was influenced by {o}.','yago:leader':'{o} led {s}.',
'yago:winner':'{o} won {s}.','yago:participant':'{o} participated in {s}.',
'yago:hasCause':'{o} caused {s}.','yago:neighbor':'{s} borders {o}.',
'yago:studentOf':'{s} was a student of {o}.','schema:worksFor':'{s} worked for {o}.',
'schema:birthPlace':'{s} was born in {o}.','yago:hasMother':'{o} is the mother of {s}.',
'yago:hasFather':'{o} is the father of {s}.','schema:deathPlace':'{s} died in {o}.',
'yago:burialPlace':'{s} was buried in {o}.','schema:spouse':'{s} was married to {o}.',
'schema:memberOf':'{s} is a member of {o}.','yago:partOf':'{s} is part of {o}.',
'yago:follows':'{s} succeeded {o}.','ex:relativeOf':'{s} and {o} are relatives.',
'ex:captiveIn':'{s} was held captive in {o}.','ex:opposed':'{s} fought against {o}.'}
# Alternative formulations must not be taken as extra independent facts.
ALT = {
'yago:creator':'{o} authored or produced {s}.', 'schema:location':'{s} lived, was based, or lay in {o}.',
'schema:about':'{s} concerns {o}.','yago:beliefSystem':'{s} adhered to {o}.',
'yago:influencedBy':'{o} influenced {s}.','yago:leader':'{s} was led by {o}.',
'yago:winner':'{o} was victorious in {s}.','yago:participant':'{o} took part in {s}.',
'yago:hasCause':'{s} happened because of {o}.','yago:neighbor':'{s} and {o} share a border.',
'yago:studentOf':'{s} studied under {o}.','schema:worksFor':'{s} served {o}.',
'schema:birthPlace':'{o} was the birthplace of {s}.','yago:hasMother':'{s} is the child of their mother {o}.',
'yago:hasFather':'{s} is the child of their father {o}.','schema:deathPlace':'{o} was where {s} died.',
'yago:burialPlace':'{s} was interred in {o}.','schema:spouse':'{s} and {o} were spouses.',
'schema:memberOf':'{s} belonged to {o}.','yago:partOf':'{s} belonged to the larger entity {o}.',
'yago:follows':'{s} followed {o} in succession.','ex:relativeOf':'{s} and {o} were related by family ties.',
'ex:captiveIn':'{s} was imprisoned or deported to {o}.','ex:opposed':'{s} rebelled or fought against {o}.'}
KIND = {'person':{'schema:Person','schema:Thing'},'location':{'schema:Place','schema:Thing'},
'place':{'schema:Place','schema:Thing'},'organization':{'schema:Organization','schema:Thing'},
'event':{'schema:Event','schema:Thing'},'written work':{'schema:CreativeWork','schema:Thing'},
'work':{'schema:CreativeWork','schema:Thing'},'religion':{'yago:BeliefSystem','schema:Thing'}}
SYMMETRIC = {'yago:neighbor','schema:spouse','ex:relativeOf'}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def norm(s):
 s=unicodedata.normalize('NFKD',str(s)).casefold()
 return re.sub(r'[^\w]+',' ', ''.join(c for c in s if not unicodedata.combining(c))).strip()
def kind(label):
 x=norm(label)
 if x in KIND:return KIND[x]
 if any(k in x for k in ('person','human','king','ruler','historian')):return KIND['person']
 if any(k in x for k in ('geograph','city','town','place','region','country','empire','state','river','mountain')):return KIND['location']
 if any(k in x for k in ('organization','institution','family','dynasty','group','army','church')):return KIND['organization']
 if 'event' in x or 'battle' in x:return KIND['event']
 return {'schema:Thing'}  # unknown: retain non-strict eligibility; avoid inventing a detailed type

def compatible(m, allowed, strict):
 types=kind(m['type'])
 return 'schema:Thing' in allowed or bool(types & set(allowed)) or (not strict and types == {'schema:Thing'})

def sentences(text):
 # Retain character spans; deterministic fallback without external NLTK models.
 starts=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-ÞŁŁ])', text)]
 out=[]
 for a,b in zip(starts,starts[1:]+[len(text)]):
  if text[a:b].strip():out.append((a,b))
 return out

def mentions(doc, source, gold):
 if source=='gold':
  ent={e['eid']:e for e in gold['entities']}
  return [{ 'para':m['para'],'start':m['start'],'end':m['end'],'text':m['text'],
    'entity':m['entity'],'type':ent[m['entity']]['type'], 'yago':ent[m['entity']].get('yago')}
   for m in gold['mentions']]
 p=ROOT/'lela/pred/llm'/f'{doc}.jsonl'
 if not p.exists():raise FileNotFoundError(p)
 payload=json.loads(p.read_text(encoding='utf8').splitlines()[0])
 return [{'para':m['para'],'start':m['start'],'end':m['end'],'text':m['text'],
 'entity':m.get('entity_id') or f"surface:{norm(m['text'])}",'type':m.get('label',''),
 'yago':m.get('entity_id')} for m in payload['entities']]

def candidate_jobs(doc, source, pair_scope, strict_types, schema):
 gold=read(ROOT/'dataset/gold'/f'{doc}.json')
 para=read(ROOT/'dataset/paragraphs'/f'{doc}.json')
 scope=set(gold['scope']['paragraphs'])
 bypara=collections.defaultdict(list)
 for m in mentions(doc,source,gold):
  if m['para'] in scope:bypara[m['para']].append(m)
 jobs=[]
 for p in para['paragraphs']:
  pid=p['id']
  if pid not in scope:continue
  text=p['text']; segs=sentences(text) if pair_scope=='sentence' else [(0,len(text))]
  for start,end in segs:
   group={}
   for m in bypara[pid]:
    if start <= m['start'] and m['end']<=end and text[m['start']:m['end']] == m['text']:
     group.setdefault(m['entity'],m)
   for a,b in itertools.combinations(group.values(),2):
    if norm(a['text'])==norm(b['text']):continue
    for s,o in ((a,b),(b,a)):
     for rel,info in schema.items():
      if rel not in BASE:continue
      if s is b and rel in SYMMETRIC:continue
      if not compatible(s,info['domain'],strict_types) or not compatible(o,info['range'],strict_types):continue
      # Restrict trivially nonsensical recursive relations as in the Jev pipeline.
      if rel in ('yago:partOf','yago:follows') and not (kind(s['type']) & kind(o['type']) - {'schema:Thing'}):continue
      if rel=='yago:partOf' and 'schema:Place' in kind(s['type']) & kind(o['type']):continue
      jobs.append({'doc':doc,'para':pid,'start':start,'end':end,'text':text[start:end], 'paragraph_text':text,
        'subj':s['text'],'obj':o['text'],'subj_entity':s['entity'],'obj_entity':o['entity'],
        'subj_yago':s['yago'],'obj_yago':o['yago'],'rel':rel})
 return jobs

def infer(model_name, jobs, strategy, batch_size, device, max_length):
 import torch
 from transformers import AutoTokenizer, AutoModelForSequenceClassification
 tokenizer=AutoTokenizer.from_pretrained(model_name)
 model=AutoModelForSequenceClassification.from_pretrained(model_name)
 if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
 model=model.to(device).eval()
 entail=next((int(i) for i,l in model.config.id2label.items() if str(l).lower()=='entailment'),None)
 if entail is None:raise ValueError(f'No entailment class in model labels: {model.config.id2label}')
 requests=[]
 for i,j in enumerate(jobs):
  templates=[BASE[j['rel']]] if strategy=='baseline' else [ALT[j['rel']]] if strategy=='alternative' else [BASE[j['rel']],ALT[j['rel']]]
  for t in templates:requests.append((i,j['text'],t.format(s=j['subj'],o=j['obj'])))
 scores=[[] for _ in jobs]
 for start in range(0,len(requests),batch_size):
  batch=requests[start:start+batch_size]
  inputs=tokenizer([x[1] for x in batch],[x[2] for x in batch],padding=True,truncation=True,
    max_length=max_length,return_tensors='pt').to(device)
  with torch.inference_mode():pr=torch.softmax(model(**inputs).logits,dim=-1)[:,entail].detach().cpu().tolist()
  for (idx,_,_),p in zip(batch,pr):scores[idx].append(float(p))
  if start==0 or (start//batch_size)%50==0:print(f'{strategy} {start+len(batch)}/{len(requests)} scored',flush=True)
 for j,ss in zip(jobs,scores):
  j['score']=max(ss) if ss else 0.0
  j['hypothesis_scores']=ss
 return jobs

def predict(jobs, threshold, multi):
 bypair=collections.defaultdict(list)
 for j in jobs:
  # identical pair independently across sentences: rank by score before dedup
  bypair[(j['doc'],j['para'],j['subj_entity'],j['obj_entity'])].append(j)
 chosen=[]
 for options in bypair.values():
  options=sorted(options,key=lambda x:-x['score'])
  if multi:chosen.extend(x for x in options if x['score']>=threshold)
  elif options and options[0]['score']>=threshold:chosen.append(options[0])
 seen=set();out=[]
 for j in sorted(chosen,key=lambda x:-x['score']):
  k=(j['doc'],j['para'],j['subj_entity'],j['rel'],j['obj_entity'])
  if k in seen:continue
  seen.add(k);out.append(j)
 return out

def evaluate(doc, preds, source):
 gold=read(ROOT/'dataset/gold'/f'{doc}.json')
 g={(f['para'],f['subj'],f['rel'],f['obj']) for f in gold['facts']}
 lookup=collections.defaultdict(set)
 for e in gold['entities']:
  for s in [e['label'],*e.get('aliases',[])]:lookup[norm(s)].add(e['eid'])
 yago=collections.defaultdict(set)
 for e in gold['entities']:
  if e.get('yago'):yago[e['yago']].add(e['eid'])
 def resolve(x,name):
  # gold source uses gold EIDs; LELA uses YAGO links with name fallback.
  if source=='gold':return x
  z=yago.get(x,set()) if x else set()
  if len(z)==1:return next(iter(z))
  z=lookup.get(norm(name),set())
  return next(iter(z)) if len(z)==1 else '__unmatched__:'+str(x)+':'+norm(name)
 p={(j['para'],resolve(j['subj_entity'],j['subj']),j['rel'],resolve(j['obj_entity'],j['obj'])) for j in preds}
 tp=len(p & g);fp=len(p-g);fn=len(g-p)
 precision=tp/len(p) if p else 0.;recall=tp/len(g) if g else 0.
 f1=2*precision*recall/(precision+recall) if precision+recall else 0.
 return {'tp':tp,'pred':len(p),'gold':len(g),'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1},p,g

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--model',default=MODEL);ap.add_argument('--batch-size',type=int,default=16)
 ap.add_argument('--device',default='auto',choices=('auto','cpu','cuda','mps'))
 ap.add_argument('--max-length',type=int,default=512)
 ap.add_argument('--threshold',type=float,default=0.9,help='Fixed threshold, not tuned on test documents')
 ap.add_argument('--sources',nargs='+',choices=('gold','lela'),default=['gold','lela'])
 ap.add_argument('--modes',nargs='+',choices=('sentence_sentence','sentence_paragraph','paragraph_paragraph'),default=['sentence_sentence','sentence_paragraph','paragraph_paragraph'], help='pair-scope_premise-scope')
 ap.add_argument('--strategies',nargs='+',choices=('baseline','alternative','ensemble'),default=['baseline','alternative','ensemble'])
 ap.add_argument('--strict-types',action='store_true',help='Reject unknown mention type; off by default for LELA')
 ap.add_argument('--multi-label',action='store_true',help='Allow >1 relation for an entity pair')
 ap.add_argument('--out',type=Path,default=HERE/'results')
 ap.add_argument('--prepare-only',action='store_true',help='Generate candidate coverage without model inference')
 args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 schema=read(ROOT/'dataset/relations.json');allresults=[];coverage=[]
 for source,mode in itertools.product(args.sources,args.modes):
  pair_scope,premise_scope=mode.split('_')
  jobs=[j for doc in DOCS for j in candidate_jobs(doc,source,pair_scope,args.strict_types,schema)]
  if premise_scope=='paragraph':
   for j in jobs:j['text']=j['paragraph_text']
  coverage.append({'source':source,'mode':mode,'candidate_relations':len(jobs),'unique_pairs':len(set((j['doc'],j['para'],j['subj_entity'],j['obj_entity']) for j in jobs))})
  path=out/f'{source}_{mode}_candidates.json'
  path.write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf8')
  if args.prepare_only:continue
  # score both distinct hypotheses once. Derive all three methods from shared scores.
  infer(args.model,jobs,'ensemble',args.batch_size,args.device,args.max_length)
  (out/f'{source}_{mode}_scores.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf8')
  for strategy in args.strategies:
   selected=[]
   for j in jobs:
    k=dict(j);ss=k['hypothesis_scores'];k['score']=ss[0] if strategy=='baseline' else ss[1] if strategy=='alternative' else max(ss)
    selected.append(k)
   pred=predict(selected,args.threshold,args.multi_label)
   run=f'{source}_{mode}_{strategy}'
   for doc in DOCS:
    facts=[j for j in pred if j['doc']==doc]
    dest=out/'pred'/run;dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{doc}.json').write_text(json.dumps({'doc_id':doc,'run':run,'model':args.model,
      'threshold':args.threshold,'facts':facts},ensure_ascii=False,indent=2),encoding='utf8')
    metrics,_,_=evaluate(doc,facts,source)
    allresults.append({'run':run,'doc':doc,**metrics})
   totals={k:sum(x[k] for x in allresults if x['run']==run) for k in ('tp','pred','gold')}
   p=totals['tp']/totals['pred'] if totals['pred'] else 0
   r=totals['tp']/totals['gold'] if totals['gold'] else 0
   allresults.append({'run':run,'doc':'ALL',**totals,'fp':totals['pred']-totals['tp'],
    'fn':totals['gold']-totals['tp'],'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0})
 covpath=out/'coverage.json'
 oldcov=read(covpath) if covpath.exists() else []
 merged={(r['source'],r['mode']):r for r in oldcov}
 merged.update({(r['source'],r['mode']):r for r in coverage})
 covpath.write_text(json.dumps(list(merged.values()),indent=2),encoding='utf8')
 if allresults:
  # Merge previously completed runs rather than overwriting their summaries.
  metrics_path=out/'metrics.csv'
  old=list(csv.DictReader(metrics_path.open(encoding='utf8'))) if metrics_path.exists() else []
  indexed={(r['run'],r['doc']):r for r in old}
  indexed.update({(r['run'],r['doc']):r for r in allresults})
  allresults=list(indexed.values())
  with metrics_path.open('w',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fieldnames=list(allresults[0]));writer.writeheader();writer.writerows(allresults)
  lines=['# Fresh NLI inference results','','Fixed threshold: '+str(args.threshold),'','| Run | Doc | TP / predicted / gold | P | R | F1 |','|---|---|---|---:|---:|---:|']
  for x in allresults:lines.append(f"| {x['run']} | {x['doc']} | {x['tp']} / {x['pred']} / {x['gold']} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} |")
  (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('Saved:',out,'inference run:',not args.prepare_only)
if __name__=='__main__':main()
