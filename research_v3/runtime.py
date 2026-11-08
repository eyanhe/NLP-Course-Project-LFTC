"""Record actual interpreter, imports and GPU arithmetic without fitting a model."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .storage import atomic_write_json

CORE_IMPORTS = {'numpy': 'numpy', 'pandas': 'pandas', 'pyarrow': 'pyarrow',
                'scikit-learn': 'sklearn', 'torch': 'torch', 'transformers': 'transformers',
                'datasets': 'datasets', 'gensim': 'gensim', 'datasketch': 'datasketch',
                'psutil': 'psutil', 'safetensors': 'safetensors', 'huggingface-hub': 'huggingface_hub'}


def probe_environment(output: Path) -> dict[str, Any]:
    versions, errors = {}, {}
    for package, module in CORE_IMPORTS.items():
        try:
            versions[package] = importlib.metadata.version(package)
            importlib.import_module(module)
        except Exception as exc:
            errors[package] = f'{type(exc).__name__}: {exc}'
    evidence: dict[str, Any] = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'python_executable': sys.executable,
        'virtual_environment': sys.prefix, 'platform': platform.platform(),
        'logical_cpus': os.cpu_count(), 'packages': versions, 'import_errors': errors,
        'actual_model_training_run': False, 'gpu': {'available': False},
        'g0_full_training_smoke': 'not_run_user_requested_pretraining_only',
    }
    if 'psutil' not in errors:
        import psutil
        evidence['system_memory_bytes'] = psutil.virtual_memory().total
        evidence['process_rss_bytes'] = psutil.Process().memory_info().rss
    if 'torch' not in errors:
        import torch
        torch.set_num_threads(4)
        evidence['torch_cuda_runtime'] = torch.version.cuda
        if torch.cuda.is_available():
            try:
                properties = torch.cuda.get_device_properties(0)
                with torch.inference_mode():
                    a = torch.ones((16, 16), device='cuda')
                    value = float((a @ a)[0, 0].cpu())
                evidence['gpu'] = {'available': True, 'name': properties.name,
                                   'total_memory_bytes': properties.total_memory,
                                   'arithmetic_verified': value == 16.0,
                                   'bf16_supported': torch.cuda.is_bf16_supported()}
                del a
                torch.cuda.empty_cache()
            except Exception as exc:
                evidence['gpu'] = {'available': False, 'error': repr(exc)}
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        completed = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                    '--format=csv,noheader'], capture_output=True, text=True,
                                   timeout=20, creationflags=flags)
        evidence['nvidia_smi'] = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence['nvidia_smi_error'] = str(exc)
    evidence['runtime_ready'] = not errors and sys.version_info[:2] == (3, 11)
    atomic_write_json(Path(output), evidence, Path(output).parent)
    return evidence
