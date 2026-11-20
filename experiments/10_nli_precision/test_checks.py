import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from checks import choose,score_strategy
from focused_inference import mark_entities

class Checks(unittest.TestCase):
 def jobs(self):
  return [dict(doc='d',para='p',subj_entity='a',obj_entity='b',rel=rel,score=score) for rel,score in [('r1',.96),('r2',.94)]]
 def test_margin_rejects_ambiguous(self):self.assertEqual(choose(self.jobs(),.9,.05),[])
 def test_margin_accepts(self):self.assertEqual(len(choose(self.jobs(),.9,.01)),1)
 def test_threshold(self):self.assertEqual(choose(self.jobs(),.97,0),[])
 def test_two_hypotheses(self):
  v=score_strategy([{'hypothesis_scores':[.4,.8]}],'ensemble')
  self.assertEqual(v[0]['score'],.8)
 def test_marking(self):
  s=mark_entities('Plato travelled to Athens.','Plato','Athens')
  self.assertIn('[SUBJ] Plato [/SUBJ]',s)
  self.assertIn('[OBJ] Athens [/OBJ]',s)
 def test_repeated_names(self):
  s=mark_entities('Plato met Plato in Athens.','Plato','Athens')
  self.assertEqual(s.count('[SUBJ]'),1)
 def test_overlapping(self):self.assertIsNone(mark_entities('New York','New York','York'))
 def test_missing(self):self.assertIsNone(mark_entities('Plato spoke.','Plato','Athens'))
if __name__=='__main__':unittest.main()
