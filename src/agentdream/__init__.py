"""agentdream — local-first sleep-time memory consolidation ("dreaming") for any agent and any model.

    from agentdream import dream
    job = dream(memory="./mem", sessions=["./logs"], backend="deterministic")
    print(job.report["counts"])
"""

__version__ = "0.1.0"

from .adapters import MemoryAdapter, available_adapters, get_adapter, register_adapter
from .api import dream, dream_async, make_config
from .backends import LLMBackend, available_backends, get_backend, register_backend
from .context import DreamCanceled, DreamConfig, DreamContext
from .jobs import DreamJob, JobNotFound
from .loaders import load_transcript, load_transcripts, register_loader
from .models import Evidence, MemoryEntry, Message, Operation, Transcript, apply_operations
from .pipeline import DreamLimitError, Pipeline, run_dream
from .redact import Redactor, redact
from .review import AcceptResult, ReviewError, accept, discard, restore
from .scheduler import Scheduler

__all__ = [
    "AcceptResult",
    "DreamCanceled",
    "DreamConfig",
    "DreamContext",
    "DreamJob",
    "DreamLimitError",
    "Evidence",
    "JobNotFound",
    "LLMBackend",
    "MemoryAdapter",
    "MemoryEntry",
    "Message",
    "Operation",
    "Pipeline",
    "Redactor",
    "ReviewError",
    "Scheduler",
    "Transcript",
    "__version__",
    "accept",
    "apply_operations",
    "available_adapters",
    "available_backends",
    "discard",
    "dream",
    "dream_async",
    "get_adapter",
    "get_backend",
    "load_transcript",
    "load_transcripts",
    "make_config",
    "redact",
    "register_adapter",
    "register_backend",
    "register_loader",
    "restore",
    "run_dream",
]
