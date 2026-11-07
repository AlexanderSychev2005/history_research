#!/usr/bin/env python3
"""Convert history_research NER, linked and OpenIE JSON to eval/evaluate.py schema."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ner', required=True)
    ap.add_argument('--linked', required=True)
    ap.add_argument('--facts', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ner, linked, facts = load(a.ner), load(a.linked), load(a.facts)
    doc = facts['doc_id']
    assert ner['doc_id'] == linked['doc_id'] == doc, 'doc_id mismatch'
    entities = {}
    mentions = []
    decisions = {}
    for item in linked.get('entities', []):
        for d in item.get('decisions', []):
            key = (d.get('paragraph_id'), d.get('start'), d.get('end'))
            if d.get('status') == 'linked' and str(d.get('entity_id', '')).startswith('Q'):
                decisions[key] = d['entity_id']
    def add(eid, label):
        if eid not in entities:
            entities[eid] = {'eid': eid, 'label': label}
            if eid.startswith('Q') and eid[1:].isdigit():
                entities[eid]['qid'] = eid
    for p in ner.get('paragraphs', []):
        pid = p.get('paragraph_id', p.get('id'))
        for m in p.get('mentions', []):
            item = {'para': pid, 'text': m['text'], 'start': m.get('start'), 'end': m.get('end'), 'type': m.get('label', '')}
            eid = decisions.get((pid, m.get('start'), m.get('end')))
            if eid:
                item['entity'] = eid
                add(eid, m['text'])
            mentions.append(item)
    outfacts = []
    for p in facts.get('paragraphs', []):
        if p.get('status') != 'complete':
            continue
        pid = p['paragraph_id']
        for r in p.get('relations', []):
            subj, obj = r['subject'], r['object']
            add(subj, r.get('subject_name', subj))
            add(obj, r.get('object_name', obj))
            outfacts.append({'para': pid, 'subj': subj, 'rel': r['predicate'], 'obj': obj,
                             'evidence_quote': r.get('evidence', '')})
    result = {'doc_id': doc, 'entities': list(entities.values()), 'mentions': mentions, 'facts': outfacts}
    path = Path(a.out)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / f'{doc}.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {dest}: {len(entities)} entities, {len(mentions)} mentions, {len(outfacts)} facts')
    print('Note: only explicitly linked mentions have entity IDs; no NIL/unknown links invented.')


if __name__ == '__main__':
    main()
