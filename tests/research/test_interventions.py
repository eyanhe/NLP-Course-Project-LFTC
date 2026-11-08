"""Intervention checks must require real reviews and predeclared paired controls."""
import importlib
import unittest


class InterventionTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.interventions')
        except ModuleNotFoundError:self.fail('predeclared intervention support is missing')

    def test_m1_deletion_is_not_a_content_word_replacement(self):
        pair={'mechanism':'M1','sample_id':'fixture/D/0','group_id':'g','source_partition':'discovery',
              'original_text':'please help me transfer cash right now',
              'intervention_text':'please help me transfer right now',
              'control_text':'please help transfer cash right now','gold_labels':[0],'cue_words':['cash'],
              'reviews':[{'reviewer_id':r,'blind_to_predictions':True,'intervention_label_preserved':True,
                  'control_label_preserved':True,'control_natural':True,'intervention_content_words_replaced':True,
                  'control_content_words_replaced':True} for r in ['synthetic-A','synthetic-B']]}
        with self.assertRaisesRegex(ValueError,'replacement|substitution'):
            self.api().validate_pair(pair)

    def test_m1_needs_human_content_word_attestations_for_both_arms(self):
        pair={'mechanism':'M1','sample_id':'fixture/D/0','group_id':'g','source_partition':'discovery',
              'original_text':'I want to change my card PIN today',
              'intervention_text':'I want to update my card PIN today',
              'control_text':'I want to change my card PIN now','gold_labels':[1],'cue_words':['change'],
              'reviews':[{'reviewer_id':r,'blind_to_predictions':True,'intervention_label_preserved':True,
                  'control_label_preserved':True,'control_natural':True} for r in ['synthetic-A','synthetic-B']]}
        with self.assertRaisesRegex(ValueError,'content.word'):
            self.api().validate_pair(pair)

    def test_paired_sample_size_uses_discordance_upper_bound_and_limits(self):
        api=self.api()
        self.assertEqual(api.plan_pair_count(0,50),400)
        self.assertEqual(api.plan_pair_count(50,50),1047)
        self.assertEqual(api.plan_pair_count(50,50,family_size=100),1200)
        with self.assertRaises(ValueError):api.plan_pair_count(51,50)

    def test_m1_equal_word_edit_distance_and_two_real_review_records(self):
        pair={'mechanism':'M1','sample_id':'fixture/D/0','group_id':'g','source_partition':'discovery',
              'original_text':'I want to change my card PIN today',
              'intervention_text':'I want to update my card PIN today',
              'control_text':'I want to change my card PIN now','gold_labels':[1],
              'cue_words':['change'],'reviews':[]}
        api=self.api()
        with self.assertRaisesRegex(ValueError,'review|human'):api.validate_pair(pair)
        pair['reviews']=[{'reviewer_id':name,'blind_to_predictions':True,
                         'intervention_label_preserved':True,'control_label_preserved':True,
                         'control_natural':True,'intervention_content_words_replaced':True,
                         'control_content_words_replaced':True} for name in ['fixture_person1','fixture_person2']]
        api.validate_pair(pair)
        changed={**pair,'control_text':'I need to change the PIN immediately now'}
        with self.assertRaisesRegex(ValueError,'edit|distance|control'):api.validate_pair(changed)

    def test_m4_evidence_in_intervention_and_absent_from_nearest_stride_control(self):
        api=self.api()
        words=['word']*600;text=' '.join(words)
        offsets=[(i*5,i*5+4) for i in range(600)]
        evidence=(350*5,355*5+4)
        windows=api.build_evidence_windows(text,offsets,evidence,'fixture/D/0')
        a=windows['intervention'];b=windows['control']
        self.assertEqual(a['token_end']-a['token_start'],254)
        self.assertLessEqual(a['char_start'],evidence[0]);self.assertGreaterEqual(a['char_end'],evidence[1])
        self.assertTrue(b['char_end']<=evidence[0] or b['char_start']>=evidence[1])
        self.assertEqual(b['token_start']%32,0)
        self.assertEqual(b['token_start'],96)

    def test_m4_evidence_inside_main_prefix_is_ineligible(self):
        text=' '.join(['word']*600);offsets=[(i*5,i*5+4) for i in range(600)]
        with self.assertRaisesRegex(ValueError,'outside|254|prefix'):
            self.api().build_evidence_windows(text,offsets,(50,100),'fixture/D/0')

    def test_lexical_cues_use_only_train_documents_with_declared_frequency(self):
        from research_v3.types import VisibleInput
        texts=['change pin']*5+['lost card']*5
        rows=tuple(VisibleInput(str(i),t,0,len(t),2,2,str(i)) for i,t in enumerate(texts))
        cues=self.api().extract_cues(rows,[0]*5+[1]*5,2,top_k=1,min_count=5)
        self.assertEqual(cues[0][0]['word'],'change')
        self.assertEqual(cues[1][0]['word'],'card')

    def test_pilot_export_keeps_reviews_empty_and_has_no_model_outputs(self):
        from research_v3.types import Example,PreparedDataset,VisibleInput
        text='I want to change my card PIN today'
        row=Example('banking77','D/0',text,(0,),'train','g')
        data=PreparedDataset('banking77','single_label',('pin','other'),
                {'train':(),'select':(),'discovery':(row,),'test':(),'quarantine':()},'pool')
        inputs={'D/0':VisibleInput('D/0',text,0,len(text),9,9,'visible')}
        api=self.api()
        self.assertTrue(hasattr(api,'pilot_candidate_packets'),'blind pilot packet export is missing')
        packets=api.pilot_candidate_packets(data,inputs,'M1',7301,{0:[{'word':'change'}],1:[]})
        self.assertEqual(len(packets),1)
        self.assertEqual(packets[0]['pair_template']['reviews'],[])
        self.assertIsNone(packets[0]['pair_template']['intervention_text'])
        self.assertNotIn('prediction',packets[0])
        self.assertEqual(packets[0]['eligibility_status'],'awaiting_human_review')


if __name__=='__main__':unittest.main()
