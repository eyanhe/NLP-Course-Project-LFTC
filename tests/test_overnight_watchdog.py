import importlib.util
import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

path=Path(__file__).resolve().parents[1]/'research-v3/training/overnight/watchdog.py'
spec=importlib.util.spec_from_file_location('overnight_watchdog',path)
watchdog=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogDecisionTest(unittest.TestCase):
    def test_heartbeat_write_is_independent_of_monitored_job_replacements(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'heartbeat.json'
            watchdog.save_json(path,{'status':'monitor','iteration':1})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'status':'monitor','iteration':1})

    def snapshot(self, a=32, b=9, c=0, failed=0):
        return {'new_fit_counts':{str(i):{'complete':done,'failed':failed if i==0 else 0} for i,done in enumerate((a,b,c))}}

    def test_waits_for_p1_before_submitting_last_phase(self):
        self.assertEqual(watchdog.next_action(self.snapshot(b=8),False,self.now),'monitor')
        self.assertEqual(watchdog.next_action(self.snapshot(),False,self.now),'launch_p2')
        self.assertEqual(watchdog.next_action(self.snapshot(),True,self.now),'monitor')

    def test_failure_blocks_advancement_and_deadline_stops(self):
        self.assertEqual(watchdog.next_action(self.snapshot(failed=1),False,self.now),'failed')
        self.assertEqual(watchdog.next_action(self.snapshot(),False,datetime(2026,11,8,2,0,tzinfo=timezone.utc)),'deadline_stop')

    def test_all_completed_exits_instead_of_submitting_again(self):
        self.assertEqual(watchdog.next_action(self.snapshot(c=16),True,self.now),'complete')

    def test_authorized_extension_continues_after_original_subset(self):
        data=self.snapshot(c=16)
        data['extension']={'complete':0,'failed':0}
        self.assertEqual(watchdog.next_action(data,True,self.now,77,False),'launch_extension')
        self.assertEqual(watchdog.next_action(data,True,self.now,77,True),'monitor')
        data['extension']['complete']=77
        self.assertEqual(watchdog.next_action(data,True,self.now,77,True),'complete')

    def test_scheduler_failure_is_detected_after_process_leaves_queue(self):
        data=self.snapshot()
        data['job_accounting']=[{'job_id':'123','state':'FAILED','exit_code':'1:0'}]
        self.assertEqual(watchdog.next_action(data,False,self.now),'failed')

    def test_predicted_start_beyond_deadline_is_an_alert(self):
        jobs=[{'job_id':'123','state':'PENDING','predicted_start':'2026-11-09T00:47:37','submitted_at':'2026-11-07T11:53:34'}]
        alerts=watchdog.queue_alerts(jobs,self.now)
        self.assertIn('queue_start_after_deadline',[row['kind'] for row in alerts])

    now=datetime(2026,11,7,12,0,tzinfo=timezone.utc)


if __name__=='__main__':unittest.main()
