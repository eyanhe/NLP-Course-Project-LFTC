"""Sampling must preserve budgets, paired IDs and real independent support."""
import importlib
import unittest
from dataclasses import replace
from research_v3.types import Example,PreparedDataset


def fixture(counts=(30,30,30),multilabel=False):
    rows=tuple(Example('fixture',f'f/train/{c}/{i}','',(c,),'train',f'g{c}-{i}')
               for c,n in enumerate(counts) for i in range(n))
    return PreparedDataset('fixture','multi_label' if multilabel else 'single_label',
       tuple(f'c{i}' for i in range(len(counts))),
       {'train':rows,'select':(),'discovery':(),'test':(),'quarantine':()},'pool-hash')


class SamplingTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.sampling')
        except ModuleNotFoundError:
            self.fail('deterministic sampling is missing')

    def test_scale_prefixes_are_nested_and_full_draw_is_collapsed(self):
        api=self.api();data=fixture()
        small=api.sample_scale(data,10,2026);large=api.sample_scale(data,30,2026)
        self.assertEqual(small.train_ids,large.train_ids[:10])
        self.assertEqual(api.sample_scale(data,'full',2026).condition_id,
                         api.sample_scale(data,'full',2028).condition_id)

    def test_kshot_is_exact_and_insufficient_support_is_explicit(self):
        api=self.api();data=fixture((3,4,1))
        one=api.sample_kshot(data,1,2026)
        labels={r.sample_id:r.labels[0] for r in data.partitions['train']}
        self.assertEqual(sorted(labels[i] for i in one.train_ids),[0,1,2])
        with self.assertRaisesRegex(ValueError,'infeasible|capacity'):
            api.sample_kshot(data,2,2026)

    def test_long_tail_hamilton_exact_total_reserve_and_capacity(self):
        api=self.api()
        self.assertEqual(api.allocate_long_tail((10,10,10),8,1,(2,1,0)),(3,3,2))
        counts=api.allocate_long_tail((100,100,100),30,50,(0,1,2))
        self.assertEqual(sum(counts),30);self.assertGreaterEqual(min(counts),1)
        with self.assertRaisesRegex(ValueError,'capacity|infeasible'):
            api.allocate_long_tail((2,100,100),30,50,(0,1,2))

    def test_shared_budget_is_maximum_feasible_for_all_orders(self):
        api=self.api();capacity=(3,30,30);orders=((0,1,2),(1,2,0),(2,0,1))
        n=api.shared_tail_budget(capacity,orders,(1,10,50),cap=30)
        for order in orders:
            for rho in (1,10,50):
                self.assertEqual(sum(api.allocate_long_tail(capacity,n,rho,order)),n)
        self.assertLess(n,30)
        self.assertTrue(any(self._infeasible(api,capacity,n+1,rho,order)
                            for order in orders for rho in (1,10,50)))

    def _infeasible(self,api,capacity,n,rho,order):
        try:
            api.allocate_long_tail(capacity,n,rho,order)
            return False
        except ValueError:
            return True

    def test_multilabel_zero_alpha_reuses_scale_and_covers_all_labels(self):
        api=self.api();data=fixture((20,5,2),True)
        scale=api.sample_scale(data,10,2026)
        zero=api.sample_multilabel(data,10,0,2026)
        self.assertEqual(scale.train_ids,zero.train_ids)
        self.assertEqual(scale.metadata['sample_hash'],zero.metadata['sample_hash'])
        small=api.sample_scale(data,5,2026)
        self.assertEqual(small.train_ids,scale.train_ids[:5])
        lookup={r.sample_id:r.labels for r in data.partitions['train']}
        for alpha in (-.5,0,.5):
            condition=api.sample_multilabel(data,10,alpha,2026)
            self.assertEqual(len(condition.train_ids),len(set(condition.train_ids)))
            self.assertEqual(set(l for i in condition.train_ids for l in lookup[i]),{0,1,2})

    def test_condition_rejects_a_changed_pool(self):
        api=self.api();data=fixture()
        c=api.sample_scale(data,10,2026)
        with self.assertRaisesRegex(ValueError,'pool|manifest'):
            api.validate_condition_pool(c,'changed-pool')

    def test_official_registry_multilabel_task_keeps_coverage_and_e4(self):
        data=replace(fixture((20,5,2),True),task='multilabel')
        api=self.api()
        condition=api.sample_multilabel(data,10,-.5,2026)
        self.assertIn('cooccurrence_counts',condition.metadata)
        self.assertEqual(condition.metadata['missing_training_labels'],[])


if __name__=='__main__':
    unittest.main()
