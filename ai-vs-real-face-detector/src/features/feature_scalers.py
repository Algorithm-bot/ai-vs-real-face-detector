from tqdm import tqdm
import random

def fit_physics_and_prnu_scalers(
    sample_paths: Sequence[str],
    max_samples: int = 300,
    seed: int = 42,
) -> Tuple[PhysicsNormalizer, PhysicsNormalizer]:
    """
    Fit z-score scalers for physics and PRNU vectors on a subsample of
    training paths (not the full training set -- fitting scalers on
    every image is unnecessary and expensive; a representative sample
    of max_samples is sufficient for stable mean/std estimates).
    Raises if any feature extraction fails.
    """
    paths = list(sample_paths)
    if len(paths) > max_samples:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_samples)

    physics_extractor = PhysicsFeatureExtractor()
    prnu_extractor = PRNUExtractor()
    physics_vectors: List[np.ndarray] = []
    prnu_vectors: List[np.ndarray] = []

    try:
        for path in tqdm(paths, desc="Fitting feature scalers"):
            rgb = _load_rgb(path)
            physics_vectors.append(physics_extractor.extract(rgb).vector)
            prnu_vectors.append(prnu_extractor.extract(rgb).vector)
    finally:
        physics_extractor.close()

    if not physics_vectors:
        raise RuntimeError("No training samples available to fit feature scalers.")

    physics_norm = PhysicsNormalizer.fit(np.stack(physics_vectors, axis=0))
    prnu_norm = PhysicsNormalizer.fit(
        np.stack(prnu_vectors, axis=0),
        feature_names=list(PRNU_FEATURE_NAMES),
    )
    return physics_norm, prnu_norm