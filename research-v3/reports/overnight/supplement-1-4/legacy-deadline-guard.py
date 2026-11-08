"""Reuse legacy monitoring/deadline handling, but disable both continuation actions."""
import importlib.util
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('legacy_monitor',PROJECT/'research-v3/training/overnight/watchdog.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.next_action
def monitor_without_new_submissions(*args,**kwargs):
    action=original(*args,**kwargs)
    return 'monitor' if action in {'launch_p2','launch_extension'} else action
module.next_action=monitor_without_new_submissions
module.main()
