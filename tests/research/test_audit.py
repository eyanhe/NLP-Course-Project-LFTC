"""Probability audit sampling and human review evidence must remain honest."""
import importlib
import unittest
from research_v3.types import Example,PreparedDataset


class AuditTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.audit')
        except ModuleNotFoundError:self.fail('independent failure audit support is missing')

    def data(self):
        rows=tuple(Example('banking77',f'D/{i}',f'original text {i}',(i%2,),'train',
                           'small' if i==0 else 'large') for i in range(4))
        return PreparedDataset('banking77','single_label',('a','b'),
            {'train':(),'select':(),'discovery':rows,'test':(),'quarantine':()},'pool')

    def test_group_uniform_then_row_uniform_has_correct_inclusion_weights(self):
        records=self.api().draw_audit(self.data(),'discovery',2,7301,set())
        self.assertEqual(len(records),2)
        by_group={r['group_id']:r for r in records}
        self.assertEqual(by_group['small']['inclusion_probability'],1)
        self.assertAlmostEqual(by_group['large']['inclusion_probability'],1/3)
        self.assertEqual(by_group['large']['analysis_weight'],3)
        self.assertTrue(all('model_id' not in r and 'prediction' not in r for r in records))
        self.assertTrue(all(r['annotation_template']['review_status']=='unreviewed' for r in records))

    def test_excluded_groups_never_enter_discovery_packet(self):
        rows=self.api().draw_audit(self.data(),'discovery',100,7301,{'large'})
        self.assertEqual([r['sample_id'] for r in rows],['D/0'])

    def test_empty_or_single_reviewer_cannot_claim_two_human_reviews(self):
        api=self.api();template=api.draw_audit(self.data(),'discovery',1,7301,set())[0]['annotation_template']
        with self.assertRaisesRegex(ValueError,'reviewer|human'):
            api.validate_reviews(({**template,'review_status':'double_review'},))
        single={**template,'reviewer_id':'fixture_person_1','review_status':'single_review'}
        self.assertFalse(api.validate_reviews((single,))['double_review_complete'])
        with self.assertRaisesRegex(ValueError,'two|reviewer|double'):
            api.validate_reviews(({**single,'review_status':'double_review'},))

    def test_conditional_failure_rates_use_distinct_denominators(self):
        records=[{'sample_id':str(i),'error':i in {0,2},'phenomena':['negation'] if i in {0,1,3} else [],
                  'analysis_weight':1} for i in range(5)]
        rates=self.api().phenomenon_rates(records)['negation']
        self.assertAlmostEqual(rates['p_error_given_phenomenon'],1/3)
        self.assertAlmostEqual(rates['p_phenomenon_given_error'],1/2)


if __name__=='__main__':unittest.main()
