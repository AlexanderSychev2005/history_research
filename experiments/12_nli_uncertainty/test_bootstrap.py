import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paired_bootstrap import f1,quantile,paired_bootstrap

class TestBootstrap(unittest.TestCase):
 def test_f1(self):
  self.assertAlmostEqual(f1((2,4,4)),.5)
  self.assertEqual(f1((0,0,0)),1)
 def test_quantile(self):
  self.assertAlmostEqual(quantile([0,1,2,3,4],.25),1)
 def test_determinism(self):
  u=[{'doc':'a','para':'1','original':(1,2,2),'contrast':(1,1,2)},
     {'doc':'a','para':'2','original':(0,1,1),'contrast':(0,0,1)},
     {'doc':'b','para':'1','original':(1,1,1),'contrast':(1,1,1)}]
  a=paired_bootstrap(u,200,17); b=paired_bootstrap(u,200,17)
  self.assertEqual(a,b)
  self.assertAlmostEqual(a['delta_f1'],f1((2,2,4))-f1((2,4,4)))
 def test_cannot_use_single_replicate(self):
  with self.assertRaises(ValueError):paired_bootstrap([{'doc':'a','original':(1,1,1),'contrast':(1,1,1)}],1)

if __name__=='__main__':unittest.main()
