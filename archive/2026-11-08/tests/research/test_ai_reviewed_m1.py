import copy
import unittest
import numpy as np


class AIReviewedM1Tests(unittest.TestCase):
    def example(self):
        pair=dict(mechanism='M1',source_partition='discovery',sample_id='bank/1',group_id='group1',gold_labels=[0],
            original_text='Please explain your exchange rate policy.',cue_words=['exchange','rate'],
            intervention_text='Please explain your conversion rate policy.',
            control_text='Please describe your exchange rate policy.')
        mapping=[dict(case_id='M1-001',sample_id='bank/1',candidate_order=0)]
        reports=[dict(reviewer=x,records=[dict(caseId='M1-001',sampleId='bank/1',decisions=['是']*7)]) for x in ['A','B']]
        return reports,[pair],mapping

    def test_ai_acceptance_does_not_fabricate_human_review_records(self):
        from research_v3.ai_reviewed_m1 import select_ai_subset
        reports,pairs,mapping=self.example();before=copy.deepcopy(pairs)
        selected,rejected=select_ai_subset(reports,pairs,mapping)
        self.assertEqual(len(selected),1);self.assertFalse(rejected)
        self.assertEqual(selected[0]['review_source'],'Gemini_AI_assisted')
        self.assertFalse(selected[0]['independent_human_reviews'])
        self.assertEqual(pairs,before);self.assertNotIn('reviews',selected[0]['pair'])
        reports[1]['records'][0]['decisions'][1]='不确定'
        self.assertEqual(len(select_ai_subset(reports,pairs,mapping)[0]),0)

    def test_source_ids_duplicates_and_edit_movement_fail_closed(self):
        from research_v3.ai_reviewed_m1 import select_ai_subset,validate_structure
        reports,pairs,mapping=self.example()
        reports[1]['records'][0]['sampleId']='other'
        with self.assertRaises(ValueError):select_ai_subset(reports,pairs,mapping)
        reports,pairs,mapping=self.example();reports[0]['records']*=2
        with self.assertRaises(ValueError):select_ai_subset(reports,pairs,mapping)
        pair=copy.deepcopy(pairs[0]);pair['control_text']='Explain please your exchange rate policy.'
        with self.assertRaises(ValueError):validate_structure(pair)
        pair=copy.deepcopy(pairs[0]);pair['control_text']='Please explain your exchange fee policy.'
        with self.assertRaises(ValueError):validate_structure(pair)

    def test_contrast_distinguishes_existing_errors_from_introduced_errors(self):
        from research_v3.ai_reviewed_m1 import summarize_contrast
        gold=np.array([0,0,0])
        original=np.array([[2.,0.],[2.,0.],[0.,2.]])
        cue=np.array([[0.,2.],[2.,0.],[0.,2.]])
        control=np.array([[2.,0.],[2.,0.],[0.,2.]])
        result=summarize_contrast(gold,original,cue,control)
        self.assertEqual(result['original_correct'],2)
        self.assertEqual(result['cue_introduced_errors'],1)
        self.assertEqual(result['control_introduced_errors'],0)
        self.assertEqual(result['introduced_error_denominator'],2)
        self.assertEqual(result['cue_introduced_error_rate'],.5)
        self.assertEqual(result['control_correct_cue_wrong'],1)
        with self.assertRaises(ValueError):summarize_contrast(gold,original[:2],cue,control)


if __name__=='__main__':unittest.main()
