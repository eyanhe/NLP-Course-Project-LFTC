"""Observed process RSS and Torch allocated/reserved peaks, with sampling scope."""
from __future__ import annotations

import threading
import time
import psutil


class ResourceSampler:
    def __init__(self,device='cpu',interval=.5):
        self.device=device;self.interval=interval;self.stop=threading.Event();self.peak=0;self.count=0

    def _sample(self):
        self.peak=max(self.peak,psutil.Process().memory_info().rss);self.count+=1

    def _loop(self):
        while not self.stop.wait(self.interval):self._sample()

    def __enter__(self):
        import torch
        self.started=time.monotonic();self.cuda=self.device.startswith('cuda') and torch.cuda.is_available()
        if self.cuda:torch.cuda.reset_peak_memory_stats()
        self._sample();self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        import torch
        self.stop.set();self.thread.join();self._sample()
        self.report={'peak_rss_sampled_bytes':self.peak,'sample_count':self.count,
            'rss_sample_interval_seconds':self.interval,'elapsed_seconds':time.monotonic()-self.started,
            'torch_cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if self.cuda else 0,
            'torch_cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if self.cuda else 0,
            'gpu_name':torch.cuda.get_device_name() if self.cuda else None,
            'note':'sampled RSS maximum; allocated/reserved Torch memory is not total device usage'}
