"""Small synthetic cases with hand-computed answers.  Run from eval/:  python -m unittest discover -s tests"""
import csv, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate as ev

TEXT = {"d": {"p1": "Vardan fought Yazdgerd at Awarayr. Vardan died.", "p2": "Nothing here."}}
GOLD = {"doc_id": "d", "scope": {"paragraphs": ["p1"]},
        "entities": [{"eid": "E1", "label": "Vardan Mamikonean", "aliases": ["Vardan"], "qid": "Q1"},
                     {"eid": "E2", "label": "Yazdgerd II", "aliases": ["Yazdgerd"]}, {"eid": "E3", "label": "Awarayr"}],
        "mentions": [{"para": "p1", "start": 0, "end": 6, "text": "Vardan", "entity": "E1"},
                     {"para": "p1", "start": 14, "end": 22, "text": "Yazdgerd", "entity": "E2"},
                     {"para": "p1", "start": 26, "end": 33, "text": "Awarayr", "entity": None},   # NIL
                     {"para": "p1", "start": 35, "end": 41, "text": "Vardan", "entity": "E1"}],
        "facts": [{"para": "p1", "subj": "E1", "rel": "foughtAgainst", "obj": "E2"}]}
GOOD = {"entities": [{"eid": "x1", "label": "Vardan", "qid": "Q1"}, {"eid": "x2", "label": "Yazdgerd II"}],
        "mentions": [dict(m, entity={"E1": "x1", "E2": "x2", None: None}[m["entity"]]) for m in GOLD["mentions"]],
        "facts": [{"para": "p1", "subj": "x1", "rel": "foughtAgainst", "obj": "x2"}]}


def run(pred, strict=True, info=None):
    return ev.evaluate(TEXT, {"d": GOLD}, {"d": pred}, strict, info or {}, 20)["overall"]


class Scores(unittest.TestCase):
    def test_perfect(self):
        r = run(GOOD)
        for k in ev.LEVELS: self.assertEqual(r[k]["F1"], 1.0, k)

    def test_empty_prediction(self):
        r = run({})
        for k in ev.LEVELS: self.assertEqual((r[k]["P"], r[k]["R"]), (0.0, 0.0), k)

    def test_missing_mention_costs_recall_only(self):
        r = run({**GOOD, "mentions": GOOD["mentions"][:3]})
        self.assertEqual(r["mention"]["P"], 1.0); self.assertAlmostEqual(r["mention"]["R"], 0.75)

    def test_extra_mention_costs_precision_only(self):
        extra = {"para": "p1", "start": 7, "end": 13, "text": "fought", "entity": None}
        r = run({**GOOD, "mentions": GOOD["mentions"] + [extra]})
        self.assertEqual(r["mention"]["R"], 1.0); self.assertAlmostEqual(r["mention"]["P"], 4 / 5)

    def test_strict_versus_overlap(self):
        short = dict(GOOD["mentions"][0], start=0, end=5, text="Varda")
        p = {**GOOD, "mentions": [short] + GOOD["mentions"][1:]}
        self.assertAlmostEqual(run(p, True)["mention"]["R"], 3 / 4)
        self.assertEqual(run(p, False)["mention"]["R"], 1.0)

    def test_wrong_entity_hurts_link_not_mention(self):
        r = run({**GOOD, "mentions": [dict(GOOD["mentions"][0], entity="x2")] + GOOD["mentions"][1:]})
        self.assertEqual(r["mention"]["F1"], 1.0); self.assertAlmostEqual(r["link"]["R"], 2 / 3)

    def test_linking_a_nil_mention_is_a_false_positive(self):
        r = run({**GOOD, "mentions": GOOD["mentions"][:2] + [dict(GOOD["mentions"][2], entity="x1")] + GOOD["mentions"][3:]})
        self.assertAlmostEqual(r["link"]["P"], 3 / 4); self.assertEqual(r["link"]["R"], 1.0)

    def test_pair_level_ignores_relation_and_direction(self):
        r = run({**GOOD, "facts": [{"para": "p1", "subj": "x2", "rel": "attacked", "obj": "x1"}]})
        self.assertEqual(r["fact"]["F1"], 0.0); self.assertEqual(r["fact_pair"]["F1"], 1.0)

    def test_rel_map(self):
        r = run({**GOOD, "facts": [{"para": "p1", "subj": "x1", "rel": "FIGHT", "obj": "x2"}]}, info={"rel_map": {"FIGHT": "foughtAgainst"}})
        self.assertEqual(r["fact"]["F1"], 1.0)

    def test_literal_objects(self):
        g = {**GOLD, "facts": [{"para": "p1", "subj": "E1", "rel": "died", "obj": {"literal": "AD 451"}}]}
        p = {**GOOD, "facts": [{"para": "p1", "subj": "x1", "rel": "died", "obj": {"literal": "ad 451"}}]}
        self.assertEqual(ev.evaluate(TEXT, {"d": g}, {"d": p}, True, {}, 5)["overall"]["fact"]["F1"], 1.0)

    def test_predictions_outside_scope_are_ignored(self):
        extra = {"para": "p2", "start": 0, "end": 7, "text": "Nothing", "entity": None}
        self.assertEqual(run({**GOOD, "mentions": GOOD["mentions"] + [extra]})["mention"]["P"], 1.0)

    def test_mentions_without_offsets_are_anchored_in_order(self):
        bare = [{k: v for k, v in m.items() if k not in ("start", "end")} for m in GOOD["mentions"]]
        r = run({**GOOD, "mentions": bare})
        self.assertEqual(r["mention"]["F1"], 1.0); self.assertEqual(r["anchored"], 4)


class Parts(unittest.TestCase):
    def test_align_by_qid_then_unique_label(self):
        a = ev.align(GOLD["entities"], [{"eid": "z", "label": "Yazdgerd"}, {"eid": "q", "label": "x", "qid": "Q1"}, {"eid": "n", "label": "Nobody"}])
        self.assertEqual(a, {"z": "E2", "q": "E1", "n": None})

    def test_ambiguous_label_does_not_align(self):
        ents = [{"eid": "A", "label": "Paul", "aliases": []}, {"eid": "B", "label": "Paul the Apostle", "aliases": ["Paul"]}]
        self.assertIsNone(ev.align(ents, [{"eid": "x", "label": "Paul"}])["x"])

    def test_normalisation(self):
        self.assertEqual(ev.norm("Ełišē"), ev.norm("elise")); self.assertEqual(ev.norm("Gorgān"), ev.norm("GORGAN"))

    def test_prf(self):
        self.assertEqual(ev.prf(0, 0, 0), (1.0, 1.0, 1.0)); self.assertEqual(ev.prf(0, 0, 3), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(ev.prf(2, 3, 4)[2], 4 / 7)   # P=2/3, R=1/2 -> F1=4/7

    def test_wilson_matches_textbook_value(self):
        lo, hi = ev.wilson(90, 100)
        self.assertAlmostEqual(lo, 0.826, places=2); self.assertAlmostEqual(hi, 0.945, places=2)
        self.assertEqual(ev.wilson(0, 0), (0.0, 1.0))

    def test_validate_flags_bad_files(self):
        bad = {"entities": [], "mentions": [{"para": "p1", "start": 0, "end": 6, "text": "XXXXXX", "entity": "nope"}],
               "facts": [{"para": "p1", "subj": "a", "rel": "r", "obj": "b", "evidence_quote": "invented"}]}
        issues, st = ev.check(bad, TEXT, "d")
        self.assertEqual(len(issues), 4); self.assertEqual(st["verbatim"], 0)

    def test_sample_and_audit_round_trip(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); (d / "texts").mkdir(); (d / "pred").mkdir()
            import json
            (d / "texts" / "d.json").write_text(json.dumps({"doc_id": "d", "paragraphs": [{"id": "p1", "text": TEXT["d"]["p1"]}]}))
            (d / "pred" / "d.json").write_text(json.dumps(GOOD | {"doc_id": "d"}))
            out = d / "audit.csv"
            ev.main(["sample", "--texts", str(d / "texts"), "--pred", str(d / "pred"), "--n", "5", "--out", str(out)])
            with open(out, encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["verdict"], "")
            rows[0]["verdict"] = "1"
            with open(out, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
            self.assertEqual(ev.main(["audit", "--csv", str(out)]), 0)

    def test_bootstrap_is_deterministic(self):
        u = [{"mention_tp": 1, "mention_pred": 1, "mention_gold": 1}, {"mention_tp": 0, "mention_pred": 1, "mention_gold": 1},
             {"mention_tp": 1, "mention_pred": 1, "mention_gold": 2}]
        self.assertEqual(ev.bootstrap(u, 50, 1), ev.bootstrap(u, 50, 1))


if __name__ == "__main__":
    unittest.main()
