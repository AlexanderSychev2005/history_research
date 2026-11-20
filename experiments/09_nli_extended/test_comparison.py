import importlib.util
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('comparison_test_module',HERE/'compare_all.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ComparisonTests(unittest.TestCase):
 def test_gold_entities_keep_eids(self):
  g={'entities':[{'eid':'X','label':'X','yago':'yago:X'}]}
  r=mod.normalize_eids('a',[],g,'gold')
  self.assertEqual(r[0]['eid'],'X')
  self.assertEqual(r[0]['qid'],'yago:X')
 def test_lela_entities_derive_yago(self):
  j={'doc':'a','para':'p','subj':'A','obj':'B','subj_entity':'yago:A','obj_entity':'surface:B','subj_yago':'yago:A','obj_yago':None}
  r=mod.normalize_eids('a',[j],{'entities':[]},'lela')
  self.assertEqual(r[0]['qid'],'yago:A')
  self.assertIsNone(r[1]['qid'])
 def test_converted_facts(self):
  j={'doc':'a','para':'p','subj':'A','obj':'B','subj_entity':'A','obj_entity':'B','rel':'schema:location'}
  g={'entities':[{'eid':'A','label':'A'},{'eid':'B','label':'B'}]}
  self.assertEqual(mod.convert_nli('a',[j],[j],g,'gold','r')['facts'],[{'para':'p','subj':'A','rel':'schema:location','obj':'B'}])

if __name__=='__main__':unittest.main()
