"""Windows progress readers must not abort the model-training queue."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from research_v3 import parallel_training
from research_v3.storage import atomic_write_json, read_json


@unittest.skipUnless(os.name == 'nt', 'exercises an actual Windows file handle')
class ProgressWriteTests(unittest.TestCase):
    def reader(self, path):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
            wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return lambda: kernel.CloseHandle(handle)

    def test_a_short_lived_progress_reader_does_not_abort_atomic_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'progress.json'
            atomic_write_json(path, {'completed_fits': 59}, root)
            close = self.reader(path)
            timer = threading.Timer(.15, close)
            timer.start()
            try:
                try:
                    parallel_training.atomic_write_json(path, {'completed_fits': 60}, root)
                except PermissionError as exc:
                    self.fail(f'A temporary Windows reader aborted progress: {exc}')
            finally:
                timer.join()
            self.assertEqual(read_json(path), {'completed_fits': 60})
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_persistent_access_failure_remains_an_error_and_preserves_old_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'progress.json'
            atomic_write_json(path, {'completed_fits': 59}, root)
            close = self.reader(path)
            started = time.monotonic()
            try:
                with self.assertRaises(PermissionError):
                    parallel_training.atomic_write_json(path, {'completed_fits': 60}, root)
            finally:
                close()
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(read_json(path), {'completed_fits': 59})
            self.assertEqual(list(root.glob('*.tmp')), [])
