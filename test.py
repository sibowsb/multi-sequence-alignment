import unittest
from greedy_aligner import *


class TestSeq2Seq(unittest.TestCase):
    def test_simple(self):
        seqs = ['ACTTGAC', 'ACGTGC']
        res = align_global(*seqs)
        expected_res = ('ACTTGAC', 'ACGT-GC', 3.0)
        self.assertEqual(res, expected_res)

    def test_edit_distance(self):
        seqs = ['ACTTGAC', 'ACGTGC']
        res = align_global(*seqs, delta=lambda x, y: int(x != y), 
                           minimize=True)
        expected_res = ('ACTTGAC', 'ACGT-GC', 2.0)
        self.assertEqual(res, expected_res)


class TestProfile(unittest.TestCase):
    def test_init(self):
        p = Profile(['GTCTGA', 'GTCAGC'])
        print(p)

    def test_profile_to_seq(self):
        p = Profile(['GTCTGA', 'GTCAGC'])
        score = p.add_str('GATTCA')
        print(p)
        expected_score = -1.0
        self.assertEqual(score, expected_score)

    def test_profile_to_profile(self):
        p1 = Profile(['GTCTGA', 'GTCAGC'])
        p2 = Profile(['GA-TTCA', 'GATAT-T'])    # try different lengths
        score = p1.add_profile(p2)
        print(p1)
        expected_score = -1.5
        self.assertEqual(score, expected_score)
        

if __name__ == '__main__':
    unittest.main()

