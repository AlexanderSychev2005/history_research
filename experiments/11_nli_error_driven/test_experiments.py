import unittest
from diagnose import classify
from reverse_probe import selected_for_probe,reversed_jobs,key
from evaluate_reverse import filter_predictions
class ExperimentsTest(unittest.TestCase):
    def candidate(self,rel='yago:leader',scores=(.95,.91)):
        return {'doc':'alford2026','para':'p001','subj':'A','obj':'B','subj_entity':'A','obj_entity':'B',
                'rel':rel,'text':'A and B','hypothesis_scores':list(scores),'score':max(scores)}
    def test_taxonomy_wrong_relation(self):
        k=('p001','A','yago:leader','B');g={('p001','A','yago:participant','B')}
        self.assertEqual(classify(k,g),'wrong_relation_same_directed_pair')
    def test_taxonomy_reversed(self):
        k=('p001','A','yago:leader','B');g={('p001','B','yago:leader','A')}
        self.assertEqual(classify(k,g),'reversed_roles_same_relation')
    def test_taxonomy_unknown(self):
        self.assertEqual(classify(('p001','?unknown','rel','B'),set()),'unmatched_entity')
    def test_reversed_jobs_does_not_mutate(self):
        c=self.candidate();r=reversed_jobs([c])[0]
        self.assertEqual((r['subj'],r['obj']),('B','A'))
        self.assertEqual((r['subj_entity'],r['obj_entity']),('A','B'))
        self.assertEqual(c['subj'],'A')
        self.assertNotIn('hypothesis_scores',r)
    def test_delta_rule_keeps_forward(self):
        c=self.candidate();paired=[{'key':list(key(c)),'reverse_scores':[.1,.2]}]
        self.assertEqual(len(filter_predictions([c],paired,.2,'ensemble')),1)
    def test_delta_rule_rejects_ambiguous(self):
        c=self.candidate();paired=[{'key':list(key(c)),'reverse_scores':[.92,.94]}]
        self.assertEqual(len(filter_predictions([c],paired,.2,'ensemble')),0)
    def test_symmetric_skips_probe(self):
        c=self.candidate('schema:spouse');self.assertEqual(len(filter_predictions([c],[],.5,'ensemble')),1)
    def test_selects_only_asymmetric(self):
        a=self.candidate();b=self.candidate('schema:spouse')
        self.assertEqual(len(selected_for_probe([a,b],'ensemble',.9)),1)
if __name__=='__main__': unittest.main()
