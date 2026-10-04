import os
import tempfile
import time
from pathlib import Path

from src.train import FaceBinaryDataset, _has_explicit_split_layout, collect_labeled_images

root = Path(tempfile.mkdtemp(prefix="explicit-layout-"))
for split in ("train", "val", "test"):
    (root / split / "real" / "src").mkdir(parents=True)
    (root / split / "fake" / "src").mkdir(parents=True)
    (root / split / "other").mkdir()
    for i in range(200):
        (root / split / "real" / "src" / f"r{i}.jpg").write_bytes(b"x")
        (root / split / "fake" / "src" / f"f{i}.jpg").write_bytes(b"y")
    for i in range(2000):
        (root / split / "other" / f"skip{i}.jpg").write_bytes(b"z")

assert _has_explicit_split_layout(root)
walked = []
orig = os.walk


def tw(p, *a, **k):
    walked.append(str(Path(p)))
    return orig(p, *a, **k)


os.walk = tw
t0 = time.perf_counter()
samples = collect_labeled_images(root / "train")
t1 = time.perf_counter()
os.walk = orig
print("samples", len(samples), "sec", round(t1 - t0, 3))
print("walked names", sorted({Path(p).name for p in walked}))
assert len(samples) == 400
assert {Path(p).name for p in walked} <= {"real", "fake", "src"}

t0 = time.perf_counter()
ds = FaceBinaryDataset(str(root), split="train", seed=42, use_physics=False)
print("dataset", len(ds.samples), "sec", round(time.perf_counter() - t0, 3))
print("labels", {s[1] for s in ds.samples})
print("ok")
