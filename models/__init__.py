"""Model package init.

Keep imports minimal to avoid importing all model backends (e.g., SlowFast/MViT)
when they are not used. The actual model modules are imported lazily in
`build_model` based on `cfg.MODEL.MODEL_NAME`.
"""

from .build import MODEL_REGISTRY, build_model

