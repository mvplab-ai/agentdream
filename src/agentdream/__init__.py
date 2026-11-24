"""agentdream — local-first sleep-time memory consolidation ("dreaming") for any agent and any model.

    from agentdream import dream
    job = dream(memory="./mem", sessions=["./logs"], backend="deterministic")
    print(job.report["counts"])
"""

__version__ = "0.1.0"

from .adapters import MemoryAdapter, available_adapters, get_adapter, register_adapter  # noqa: E402
from .api import dream, dream_async, make_config  # noqa: E402
from .backends import LLMBackend, available_backends, get_backend, register_backend  # noqa: E402
from .context import DreamCanceled, DreamConfig, DreamContext  # noqa: E402
from .jobs import DreamJob, JobNotFound  # noqa: E402
from .loaders import load_transcript, load_transcripts, register_loader  # noqa: E402
from .models import Evidence, MemoryEntry, Message, Operation, Transcript, apply_operations  # noqa: E402
from .pipeline import DreamLimitError, Pipeline, run_dream  # noqa: E402
from .redact import Redactor, redact  # noqa: E402
from .review import AcceptResult, ReviewError, accept, discard  # noqa: E402
from .scheduler import Scheduler  # noqa: E402

__all__ = [
    "__version__", "dream", "dream_async", "make_config", "DreamJob", "JobNotFound", "DreamConfig", "DreamContext",
    "DreamCanceled", "DreamLimitError", "Pipeline", "run_dream", "MemoryEntry", "Message", "Transcript", "Evidence",
    "Operation", "apply_operations", "MemoryAdapter", "get_adapter", "register_adapter", "available_adapters",
    "LLMBackend", "get_backend", "register_backend", "available_backends", "load_transcript", "load_transcripts",
    "register_loader", "Redactor", "redact", "accept", "discard", "AcceptResult", "ReviewError", "Scheduler",
]
