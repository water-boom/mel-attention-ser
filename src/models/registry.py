"""Model Registry and Factory."""

from typing import Dict, List, Type
from .backbones import (
    BaseSERModel,
    CNNBase,
    CNN_SE,
    CNN_Coord,
    CNN_MHAP,
    CNN_ASP,
    Wav2Vec2_MHAP,
)

MODEL_REGISTRY: Dict[str, Type[BaseSERModel]] = {
    "cnn_base": CNNBase,
    "cnn_se": CNN_SE,
    "cnn_coord": CNN_Coord,
    "cnn_mhap": CNN_MHAP,
    "cnn_asp": CNN_ASP,
    "w2v2_mhap": Wav2Vec2_MHAP,
}


def build_model(model_name: str, num_classes: int = 8, **kwargs) -> BaseSERModel:
    """Instantiate a model by its registry name.

    Args:
        model_name: key in MODEL_REGISTRY
        num_classes: number of emotion classes (default 8)
    Returns:
        Instance of BaseSERModel
    """
    name_lower = model_name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[name_lower]
    return model_cls(num_classes=num_classes, **kwargs)


def list_models() -> List[str]:
    """Return all registered model names."""
    return list(MODEL_REGISTRY.keys())
