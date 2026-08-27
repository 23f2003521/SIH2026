"""
Lazy, thread-safe model registry.

The SAR segmenter is a ~110 MB MiT-B2 SegFormer encoder behind a U-Net decoder;
building the graph and reading the state dict costs several seconds and a few
hundred MB of RSS. Neither Streamlit (which re-executes its script top-to-bottom
on every widget interaction) nor a multi-worker web server can afford to pay
that per request.

This registry gives the whole application exactly one instance of each model:

  * loaded lazily, so a session that only touches AIS never pays for the SAR net
  * guarded by a per-model lock, so concurrent first-touches load once, not N times
  * inference serialised per model, because the torch modules are shared mutable
    state and we run them in eval/no_grad from multiple request threads
  * instrumented, so the UI can honestly display load time and memory cost
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    """One managed model: its loader, the loaded object, and its telemetry."""

    key: str
    display_name: str
    loader: Callable[[], Any]
    description: str = ""

    _obj: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _infer_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    load_seconds: Optional[float] = None
    loaded_at: Optional[float] = None
    param_count: Optional[int] = None
    error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._obj is not None

    def get(self) -> Any:
        """Return the loaded model, loading it on first use."""
        if self._obj is not None:
            return self._obj
        with self._lock:
            if self._obj is not None:          # another thread won the race
                return self._obj
            logger.info("Loading model %r ...", self.key)
            start = time.perf_counter()
            try:
                obj = self.loader()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to load model %r", self.key)
                raise
            self.load_seconds = time.perf_counter() - start
            self.loaded_at = time.time()
            self.param_count = _count_params(obj)
            self._obj = obj
            logger.info("Loaded %r in %.2fs", self.key, self.load_seconds)
            return self._obj

    def run(self, fn: Callable[[Any], Any]) -> Any:
        """Run `fn(model)` under this model's inference lock."""
        model = self.get()
        with self._infer_lock:
            return fn(model)

    def unload(self) -> None:
        with self._lock:
            self._obj = None
            self.load_seconds = None
            self.loaded_at = None
            self.param_count = None

    def status(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.display_name,
            "description": self.description,
            "loaded": self.is_loaded,
            "load_seconds": round(self.load_seconds, 3) if self.load_seconds else None,
            "parameters": self.param_count,
            "error": self.error,
        }


def _count_params(obj: Any) -> Optional[int]:
    """Parameter count for torch modules; tuples are searched for one."""
    if isinstance(obj, tuple):
        for item in obj:
            n = _count_params(item)
            if n:
                return n
        return None
    params = getattr(obj, "parameters", None)
    if callable(params):
        try:
            return sum(p.numel() for p in params())
        except Exception:
            return None
    return None


class ModelRegistry:
    """Process-wide singleton holding every managed model."""

    _instance: Optional["ModelRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._handles: Dict[str, ModelHandle] = {}

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._register_defaults()
        return cls._instance

    def register(self, handle: ModelHandle) -> ModelHandle:
        self._handles[handle.key] = handle
        return handle

    def __getitem__(self, key: str) -> ModelHandle:
        return self._handles[key]

    @property
    def handles(self) -> Dict[str, ModelHandle]:
        return dict(self._handles)

    def status(self) -> Dict[str, Any]:
        from .config import DEVICE
        return {
            "device": str(DEVICE),
            "models": [h.status() for h in self._handles.values()],
        }

    def warmup(self, *keys: str) -> Dict[str, float]:
        """Eagerly load the named models (all of them if none named)."""
        targets = keys or tuple(self._handles)
        timings: Dict[str, float] = {}
        for key in targets:
            handle = self._handles[key]
            handle.get()
            timings[key] = handle.load_seconds or 0.0
        return timings

    def _register_defaults(self) -> None:
        # Imported lazily so that importing the registry never drags torch
        # model construction in as a side effect.
        from .inference import ais as ais_mod
        from .inference import sar as sar_mod
        from .inference import trajectory as traj_mod

        self.register(ModelHandle(
            key="sar",
            display_name="SAR Oil-Spill Segmenter",
            description="U-Net + MiT-B2 encoder, 5-class Sentinel-1 segmentation.",
            loader=sar_mod.load_model,
        ))
        self.register(ModelHandle(
            key="trajectory",
            display_name="Trajectory LSTM",
            description="2-layer LSTM, predicts a vessel's next AIS position.",
            loader=traj_mod.load_model,
        ))
        self.register(ModelHandle(
            key="ais_anomaly",
            display_name="AIS Anomaly Autoencoder",
            description="11-feature autoencoder, reconstruction-error anomaly flag.",
            loader=ais_mod.load_model,
        ))


def get_registry() -> ModelRegistry:
    return ModelRegistry.instance()
