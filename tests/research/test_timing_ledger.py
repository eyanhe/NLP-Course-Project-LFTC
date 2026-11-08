import importlib.util
from pathlib import Path
import unittest


class TimingLedgerTests(unittest.TestCase):
    def api(self):
        path=Path(__file__).resolve().parents[2]/'research-v3/reports/overnight/convergence-v1/collect-timings.py'
        self.assertTrue(path.exists(),'timing ledger collector is missing')
        spec=importlib.util.spec_from_file_location('timing_ledger',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_resume_adds_interrupted_stage_and_never_uses_last_segment_as_whole(self):
        api=self.api();status={'status':'complete','total_seconds':50,'request_hash':'same'}
        leases=[{'started_at':100,'request_hash':'same'}]
        claims=[{'claimed_at_utc':'1970-01-01T00:01:39+00:00','job_id':'1'}]
        jobs={'1':{'State':'CANCELLED','End':'1970-01-01T00:02:00'}}
        total,prior,kind=api.training_wall_time(status,leases,claims,jobs)
        self.assertEqual((total,prior),(70,20))
        self.assertEqual(kind,'scheduler_lease_reconstruction')
        self.assertIsNone(api.training_wall_time(status,leases,claims,{})[0])
        self.assertIsNone(api.training_wall_time(status,[],claims,jobs)[0])
        self.assertIsNone(api.training_wall_time(status,leases,claims+[dict(claims[0])],jobs)[0])

    def test_incomplete_or_missing_timings_remain_blank(self):
        api=self.api()
        self.assertIsNone(api.training_wall_time({'status':'running'},[],[],{})[0])
        self.assertEqual(api.training_wall_time({'status':'complete','total_seconds':0},[],[],{})[0],0)


if __name__=='__main__':unittest.main()
