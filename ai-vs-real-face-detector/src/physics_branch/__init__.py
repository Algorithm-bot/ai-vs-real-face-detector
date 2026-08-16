__all__ = ["PhysicsFeatureExtractor", "PhysicsFeatureVector", "PHYSICS_FEATURE_DIM"]


def __getattr__(name: str):
    if name in __all__:
        from .feature_vector import (
            PHYSICS_FEATURE_DIM,
            PhysicsFeatureExtractor,
            PhysicsFeatureVector,
        )

        return {
            "PhysicsFeatureExtractor": PhysicsFeatureExtractor,
            "PhysicsFeatureVector": PhysicsFeatureVector,
            "PHYSICS_FEATURE_DIM": PHYSICS_FEATURE_DIM,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
