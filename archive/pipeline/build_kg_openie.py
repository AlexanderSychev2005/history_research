"""Open relation extraction -> entity linking -> RDF with paragraph provenance."""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from rdflib import Graph, Literal, Namespace, RDF, RDFS

MODEL = 'qwen/qwen3.8-27b'
VERSION = 'openie-v2'
EX = Namespace('https://example.org/history-research/')
WD = Namespace('http://www.wikidata.org/entity/')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[‘’ʼ`]', "'", s)
    return re.sub(r'[^\w]+', ' ', s).strip()


def local_id(name):
    return 'local_' + hashlib.sha1(norm(name).encode()).hexdigest()[:14]


def get_linked_by_paragraph(data):
    result = {}
    for entity in data.get('entities', []):
        for d in entity.get('decisions', []):
            if d.get('status') != 'linked' or not re.fullmatch(r'Q\d+', str(d.get('entity_id', ''))):
                continue
            pid = d.get('paragraph_id')
            if pid:
                result.setdefault(pid, []).append({'name': entity['mention'], 'id': d['entity_id']})
    return result


def resolve(name, paragraph_id, linked):
    """Conservative: only exact normalized mention match in same paragraph."""
    matches = {e['id'] for e in linked.get(paragraph_id, []) if norm(e['name']) == norm(name)}
    if len(matches) == 1:
        return next(iter(matches))
    return local_id(name)


def clean_predicate(s):
    s = norm(s).replace(' ', '_')
    return s[:90] if re.fullmatch(r'[a-z][a-z0-9_]*', s) else None


def extract(client, paragraph):
    text = paragraph['text']
    prompt = (
        'Perform OPEN relation extraction on this historical paragraph. '
        'Identify subject, predicate, object directly from the paragraph; '
        'do NOT restrict them to any preselected entity list. '
        'Subject and object must be specific named persons, groups, places, institutions, or events explicitly mentioned '
        'in the evidence (not generic pronouns). Preserve names as written in the evidence. '
        'Predicate is a short English verb phrase. Extract only factual, explicitly supported relations. '
        'Pay close attention to passive voice and who acts on whom. '
        'Do not invent an opponent, beneficiary, location, or agent. '
        'Evidence must be an EXACT contiguous substring of the paragraph containing BOTH named participants. '
        'If you cannot quote both participants in one evidence span, omit that relation. '
        'Return ONLY a compact JSON object of this form: '
        '{"relations":[{"subject":"name in evidence","predicate":"verb phrase",'
        '"object":"name in evidence","evidence":"exact substring"}]}. '
        'If none, return {"relations":[]}.\n\nParagraph:\n' + text[:5000]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'system', 'content': 'Extract evidence-grounded relations. Return valid JSON only.'},
                  {'role': 'user', 'content': prompt}],
        temperature=0,
        response_format={'type': 'json_object'},
        max_completion_tokens=1600,
        reasoning_effort='none',
    )
    raw = json.loads(response.choices[0].message.content)
    facts = []
    for item in raw.get('relations', []):
        if not isinstance(item, dict):
            continue
        subject = str(item.get('subject', '')).strip()
        obj = str(item.get('object', '')).strip()
        evidence = str(item.get('evidence', '')).strip()
        predicate = clean_predicate(item.get('predicate', ''))
        if not subject or not obj or norm(subject) == norm(obj) or not predicate:
            continue
        if not evidence or evidence not in text:
            continue
        # Strict grounding: both named participants must be explicitly in the quoted evidence.
        if norm(subject) not in norm(evidence) or norm(obj) not in norm(evidence):
            continue
        facts.append({'subject_name': subject, 'predicate': predicate,
                      'object_name': obj, 'evidence': evidence,
                      'paragraph_id': paragraph['id']})
    return facts


def build_rdf(doc_id, facts, path):
    graph = Graph()
    graph.bind('ex', EX)
    graph.bind('wd', WD)
    graph.bind('rdfs', RDFS)

    def uri(eid):
        return WD[eid] if re.fullmatch(r'Q\d+', eid) else EX[f'{doc_id}/entity/{eid}']

    for i, fact in enumerate(facts, 1):
        s, o = uri(fact['subject']), uri(fact['object'])
        p = EX['relation/' + fact['predicate']]
        graph.add((s, RDFS.label, Literal(fact['subject_name'])))
        graph.add((o, RDFS.label, Literal(fact['object_name'])))
        graph.add((s, p, o))
        statement = EX[f'{doc_id}/statement/{i}']
        for triple in [(statement, RDF.type, RDF.Statement),
                       (statement, RDF.subject, s),
                       (statement, RDF.predicate, p),
                       (statement, RDF.object, o),
                       (statement, EX.evidence, Literal(fact['evidence'])),
                       (statement, EX.paragraphId, Literal(fact['paragraph_id'])),
                       (statement, EX.sourceDocument, EX['document/' + doc_id])]:
            graph.add(triple)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(path), format='turtle')
    return len(graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--linked', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rdf', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=0, help='0 = all paragraphs')
    args = parser.parse_args()
    load_dotenv(dotenv_path='.env')
    if not os.getenv('GROQ_API_KEY'):
        parser.error('GROQ_API_KEY missing in .env')
    text_data = json.loads(args.text.read_text(encoding='utf-8'))
    linked_data = json.loads(args.linked.read_text(encoding='utf-8'))
    doc_id = text_data['doc_id']
    if linked_data['doc_id'] != doc_id:
        parser.error('doc_id mismatch between text and linked files')
    paragraphs = [{**p, 'id': p.get('id', p.get('paragraph_id'))} for p in text_data['paragraphs']]
    if args.limit > 0:
        paragraphs = paragraphs[:args.limit]
    linked = get_linked_by_paragraph(linked_data)
    old = json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {}
    if old.get('doc_id') != doc_id or old.get('extractor_version') != VERSION:
        old = {}
    results = {p['paragraph_id']: p for p in old.get('paragraphs', []) if p.get('status') == 'complete'}
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    for i, paragraph in enumerate(paragraphs, 1):
        pid = paragraph['id']
        if pid in results:
            print(f'[{i}/{len(paragraphs)}] {pid}: reused')
            continue
        try:
            extracted = extract(client, paragraph)
            facts = []
            for f in extracted:
                f['subject'] = resolve(f['subject_name'], pid, linked)
                f['object'] = resolve(f['object_name'], pid, linked)
                facts.append(f)
            results[pid] = {'paragraph_id': pid, 'status': 'complete', 'relations': facts}
            print(f'[{i}/{len(paragraphs)}] {pid}: {len(facts)} relations')
        except Exception as exc:
            results[pid] = {'paragraph_id': pid, 'status': 'error', 'error': str(exc), 'relations': []}
            print(f'[{i}/{len(paragraphs)}] {pid}: ERROR {str(exc)[:150]}')
            if '429' in str(exc):
                time.sleep(15)
        save(args.output, {'doc_id': doc_id, 'extractor_version': VERSION,
                           'paragraphs': list(results.values())})
        time.sleep(3)
    selected = [results[p['id']] for p in paragraphs if p['id'] in results]
    facts = [f for p in selected for f in p['relations']]
    count = build_rdf(doc_id, facts, args.rdf)
    print(f'Saved {len(facts)} relations -> {args.output}')
    print(f'Saved {count} RDF triples -> {args.rdf}')
    print(f'Errors: {sum(p["status"] == "error" for p in selected)}; rerun to retry')


if __name__ == '__main__':
    main()
