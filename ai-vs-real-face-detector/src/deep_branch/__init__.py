from .preprocessing import FacePreprocessor, get_train_transforms, get_val_transforms, preprocess_for_model

__all__ = [
    "FacePreprocessor",
    "get_train_transforms",
    "get_val_transforms",
    "preprocess_for_model",
    "DeepFeatureExtractor",
    "DeepClassifier",
]


def __getattr__(name: str):
    if name in ("DeepFeatureExtractor", "DeepClassifier"):
        from .feature_extractor import DeepClassifier, DeepFeatureExtractor

        return DeepFeatureExtractor if name == "DeepFeatureExtractor" else DeepClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
