from .build import MODEL_REGISTRY, build_model
from .uniformerv2 import Uniformerv2
from .uniformerv2_reid import Uniformerv2ReID
from .videomaev2_reid import VideoMAEv2ReID

try:
    from .mvitv2_reid import MViTReID  # noqa: F401
except ImportError:
    # Optional dependency (SlowFast + PyTorchVideo) not installed; skip registration.
    MViTReID = None  # type: ignore