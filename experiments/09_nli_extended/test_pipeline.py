"""Fast tests requiring only the Python standard library and the project's benchmark."""
import unittest
import run

class Tests(unittest.TestCase):
 def test_templates_cover_schema(self):
  schema=run.read(run.ROOT/'dataset/relations.json')
  available={k for k,v in schema.items() if not any(x.startswith('xsd:') for x in v['range'])}
  self.assertEqual(available,set(run.BASE))
  self.assertEqual(available,set(run.ALT))
 def test_same_pairs_different_premises(self):
  a=run.candidate_jobs('alford2026','gold','sentence',False,run.read(run.ROOT/'dataset/relations.json'))
  self.assertTrue(a)
  self.assertTrue(any(j['text']!=j['paragraph_text'] for j in a))
 def test_no_oracle_gold_leak_to_lela_mentions(self):
  g=run.read(run.ROOT/'dataset/gold/alford2026.json')
  m=run.mentions('alford2026','lela',g)
  self.assertGreater(len(m),0)
  self.assertTrue(all(x['entity'] for x in m))
 def test_all_gold_count(self):
  self.assertEqual(sum(len(run.read(run.ROOT/'dataset/gold'/f'{d}.json')['facts']) for d in run.DOCS),54)
 def test_prediction_dedupe(self):
  j={'doc':'d','para':'p','subj_entity':'s','obj_entity':'o','rel':'r','score':0.96}
  self.assertEqual(len(run.predict([j,j],0.9,False)),1)
 def test_score_perfect_gold(self):
  g=run.read(run.ROOT/'dataset/gold/alford2026.json')
  fs=[{'para':f['para'],'subj_entity':f['subj'],'obj_entity':f['obj'], 'subj':'', 'obj':'', 'rel':f['rel']} for f in g['facts']]
  score,_,_=run.evaluate('alford2026',fs,'gold')
  self.assertEqual(score['f1'],1.0)

if __name__=='__main__':unittest.main()
