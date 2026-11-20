import unittest
from shared import key
from type_constraints import categorize,passes,permits
from evidence import evidence,policy
from hard_negatives import jobs_for,filter_negative

class Tests(unittest.TestCase):
    def job(self,rel='schema:worksFor'):
        return {'doc':'alford2026','para':'p001','subj':'Anna','obj':'Boris','subj_entity':'A','obj_entity':'B',
                'rel':rel,'score':.95,'hypothesis_scores':[.91,.95],
                'text':'Anna served Boris during the war.'}
    def test_type_known(self):
        j=self.job();self.assertTrue(passes(j,{('alford2026','A'):'person',('alford2026','B'):'person'}))
        self.assertFalse(passes(j,{('alford2026','A'):'location',('alford2026','B'):'person'}))
    def test_unknown_retained(self):
        self.assertTrue(passes(self.job(),{}))
    def test_creator_reversed_template(self):
        j=self.job('yago:creator');self.assertTrue(passes(j,{('alford2026','A'):'written work',('alford2026','B'):'person'}))
        self.assertFalse(passes(j,{('alford2026','A'):'person',('alford2026','B'):'written work'}))
    def test_evidence(self):
        e=evidence(self.job());self.assertTrue(e['both_names']);self.assertTrue(e['lexical_cue']);self.assertLess(e['distance_chars'],100)
    def test_missing_mentions(self):
        j=self.job();j['text']='No names here';self.assertFalse(evidence(j)['both_names'])
    def test_hard_negatives(self):
        j=self.job();jobs=jobs_for([j]);self.assertEqual(len(jobs),1);self.assertIn('Boris',jobs[0]['hard_negative'])
        kept=filter_negative([j],[{'key':list(key(j)),'negative_score':.85}],margin=.2)
        self.assertEqual(kept,[])
        kept=filter_negative([j],[{'key':list(key(j)),'negative_score':.1}],margin=.2)
        self.assertEqual(kept,[j])
    def test_unscored_relation_unchanged(self):
        j=self.job('schema:spouse');self.assertEqual(filter_negative([j],[],margin=.2),[j])
    def test_bad_missing_score(self):
        with self.assertRaises(KeyError):filter_negative([self.job()],[],margin=.2)
if __name__=='__main__':unittest.main()
