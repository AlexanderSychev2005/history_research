"""Tests for offline semantics-aware reverse-probe filtering."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from semantics_aware import check_pairs, decisions

class SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.selected=[
            dict(doc='d',para='p',subj='A',obj='B',subj_entity='A',obj_entity='B',rel='ex:opposed',score=.98),
            dict(doc='d',para='p',subj='C',obj='D',subj_entity='C',obj_entity='D',rel='schema:about',score=.96),
            dict(doc='d',para='p',subj='E',obj='F',subj_entity='E',obj_entity='F',rel='schema:location',score=.99),
            dict(doc='d',para='p',subj='G',obj='H',subj_entity='G',obj_entity='H',rel='ex:relativeOf',score=.99),
        ]
        self.paired=[
            dict(key=['d','p','A','ex:opposed','B'],forward_score=.98,forward_hypothesis_scores=[.98,.9],reverse_scores=[.95,.8]),
            dict(key=['d','p','C','schema:about','D'],forward_score=.96,forward_hypothesis_scores=[.96,.9],reverse_scores=[.9,.88]),
            dict(key=['d','p','E','schema:location','F'],forward_score=.99,forward_hypothesis_scores=[.99,.9],reverse_scores=[.01,.02]),
        ]
    def test_coverage(self):
        self.assertEqual(len(check_pairs(self.selected,self.paired)),3)
    def test_missing_reverse_errors(self):
        with self.assertRaises(ValueError):check_pairs(self.selected,self.paired[:-1])
    def test_duplicate_reverse_errors(self):
        with self.assertRaises(ValueError):check_pairs(self.selected,self.paired+self.paired[:1])
    def test_forward_mismatch_errors(self):
        altered=[dict(x) for x in self.paired];altered[0]['forward_score']=.4
        with self.assertRaises(ValueError):check_pairs(self.selected,altered)
    def test_symmetric_untouched(self):
        kept,_=decisions(self.selected,check_pairs(self.selected,self.paired),'contrast_all',.2)
        self.assertEqual([x['rel'] for x in kept],['schema:location','ex:relativeOf'])
    def test_exempt_opposed(self):
        kept,_=decisions(self.selected,check_pairs(self.selected,self.paired),'exempt_opposed',.2)
        self.assertEqual([x['rel'] for x in kept],['ex:opposed','schema:location','ex:relativeOf'])
    def test_exempt_about(self):
        kept,_=decisions(self.selected,check_pairs(self.selected,self.paired),'exempt_about',.2)
        self.assertEqual([x['rel'] for x in kept],['schema:about','schema:location','ex:relativeOf'])
    def test_original_unchanged(self):
        kept,_=decisions(self.selected,{},'original',.2)
        self.assertEqual(len(kept),4)
    def test_invalid_policy(self):
        with self.assertRaises(ValueError):decisions(self.selected,{},'oops',.2)

if __name__=='__main__':unittest.main()
