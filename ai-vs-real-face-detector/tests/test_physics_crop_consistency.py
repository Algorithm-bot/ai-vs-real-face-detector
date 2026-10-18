import numpy as np
import torch
from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.preprocess import preprocess_for_freq
from src.train import process_face_sample

def test_physics_branch_uses_center_crop_consistency():
    # Create a random image larger than the crop size
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    # Expected physics vector when feeding the correctly cropped image directly
    expected_vec = PhysicsFeatureExtractor().extract(preprocess_for_freq(img)).vector
    # Run the dataset processing pipeline which now should use the same crop
    sample = process_face_sample(
        rgb=img,
        label=0,
        source="test",
        preprocessor=None,
        transform=lambda x: x,
        physics_extractor=PhysicsFeatureExtractor(),
        prnu_extractor=None,
        semantic_extractor=None,
        use_physics=True,
        use_prnu=False,
        use_semantic=False,
        cached=None,
        use_freq=False,
        path="",
        epoch=0,
        seed=42,
        enable_augment=False,
        preprocess_mode="crop",
        crop_size=224,
    )
    # The pipeline returns a torch tensor for physics
    obtained_vec = sample["physics"].numpy()
    assert np.allclose(obtained_vec, expected_vec), "Physics branch does not use the shared center crop"
