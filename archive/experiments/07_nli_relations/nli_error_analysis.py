"""Diagnostic error analysis for saved NLI relation scores (no model inference).
Run from repository root: python archive/experiments/07_nli_relations/nli_error_analysis.py
Matching is by normalized gold labels/aliases, not the official span-based evaluator.
"""
import csv
import json
from collections import Counter
from pathlib import Path

DOCS = ('nechaeva2017', 'alford2026')
THRESHOLD = 0.99
SOURCE = Path('archive/outputs')
GOLD = Path('archive/eval/gold')
DEST = Path('archive/experiments/07_nli_relations/nli_error_analysis.csv')


def norm(value):
    return ' '.join(value.casefold().split())


def allowed(option, types):
    if option['rel'] != 'yago:partOf':
        return True
    a = types.get(option['subj'], set())
    b = types.get(option['obj'], set())
    return ('location' in a and 'location' in b) or ('organization' in a and 'organization' in b)


def analyze(doc):
    gold = json.loads((GOLD / f'{doc}.json').read_text(encoding='utf-8'))
    source = json.loads((SOURCE / f'{doc}_relations_gold__nli_captive.json').read_text(encoding='utf-8'))
    ner = json.loads((SOURCE / f'{doc}_ner_gold.json').read_text(encoding='utf-8'))
    types = {}
    for para in ner['paragraphs']:
        for m in para['mentions']:
            types.setdefault(m['text'], set()).add(m['label'])

    entities = {e['eid']: e for e in gold['entities']}
    decisions = source['decisions']
    rows = []
    for fact in gold['facts']:
        subject, obj = entities[fact['subj']], entities[fact['obj']]
        subj_names = {norm(x) for x in [subject['label'], *subject.get('aliases', [])]}
        obj_names = {norm(x) for x in [obj['label'], *obj.get('aliases', [])]}
        matching_decisions = []
        correct = []
        for d in decisions:
            if d['paragraph_id'] != fact['para']:
                continue
            pair_names = {norm(d['a']), norm(d['b'])}
            if not (pair_names & subj_names and pair_names & obj_names):
                continue
            matching_decisions.append(d)
            for opt in d['options']:
                if (opt['rel'] == fact['rel'] and norm(opt['subj']) in subj_names
                        and norm(opt['obj']) in obj_names):
                    correct.append((d, opt))

        row = dict(doc=doc, para=fact['para'], subject=subject['label'],
                   relation=fact['rel'], object=obj['label'],
                   pair_present=bool(matching_decisions), correct_option_present=bool(correct),
                   correct_score='', best_allowed_score='', best_allowed_relation='',
                   correct_option_allowed='', reason='')
        if not matching_decisions:
            row['reason'] = 'pair_not_generated_or_alias_mismatch'
        elif not correct:
            row['reason'] = 'correct_relation_option_missing'
        else:
            # For each matching decision, reproduce filter -> argmax -> threshold.
            candidates = []
            for d, opt in correct:
                valid = [o for o in d['options'] if allowed(o, types)]
                winner = max(valid, key=lambda o: o['p']) if valid else None
                candidates.append((d, opt, winner, allowed(opt, types)))
            d, opt, winner, is_allowed = max(candidates, key=lambda x: x[1]['p'])
            row.update(correct_score=opt['p'], correct_option_allowed=is_allowed,
                       best_allowed_score=winner['p'] if winner else '',
                       best_allowed_relation=winner['rel'] if winner else '')
            any_selected = any(
                permitted and winner and winner['p'] >= THRESHOLD
                and winner['rel'] == fact['rel']
                and norm(winner['subj']) in subj_names
                and norm(winner['obj']) in obj_names
                for _, _, winner, permitted in candidates
            )
            if any_selected:
                row['reason'] = 'selected_by_saved_scores'
            elif not any(x[3] for x in candidates):
                row['reason'] = 'filtered_by_type_constraints'
            elif max(x[1]['p'] for x in candidates if x[3]) < THRESHOLD:
                row['reason'] = 'correct_score_below_threshold'
            else:
                row['reason'] = 'other_option_won'
        rows.append(row)
    return rows


def main():
    rows = [r for doc in DOCS for r in analyze(doc)]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Analyzed {len(rows)} gold facts; saved {DEST}')
    for reason, count in Counter(r['reason'] for r in rows).most_common():
        print(f'  {reason}: {count}')
    print('NOTE: diagnostic alias-based matching; counts are not official evaluator TP/FN.')


if __name__ == '__main__':
    main()
