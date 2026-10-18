import nbformat
import ast
from pathlib import Path
import os
import sys

# Set up dummy environment
os.environ["HF_TOKEN"] = "dummy"
os.environ["PYTHONUNBUFFERED"] = "1"

# We'll define the necessary variables as dummy values
project_root = Path(".")
WORKING = Path("/kaggle/working")  # This is a fake path, but we'll use it
MODELS_ROOT = WORKING / "models_v2"
FEATURE_CACHE_DIR = WORKING / "feature_cache_v2"
EVAL_ROOT = WORKING / "evaluation_v2"
log_path = WORKING / "train.log"
SEEDS = [42]
SUBSET_SEED = 42
MAX_TRAIN_IMAGES = 200_000
SEMANTIC_MODEL = "vit_large_patch14_clip_224.openai"
SEMANTIC_DIM = 768
CACHE_BRANCHES = ["semantic", "physics", "freq"]
CACHE_BRANCHES_ARG = ",".join(CACHE_BRANCHES)
PREPROCESS = "crop"
CACHE_AUGMENT = True
NUM_WORKERS = 3
SYNC_INTERVAL_S = 300
BRANCHES = ["semantic_only", "physics_only", "freq_only", "stage1", "full_hybrid"]
HPARAMS = {
    "semantic_only": dict(epochs=60, batch_size=128, lr=1e-3),
    "physics_only": dict(epochs=60, batch_size=128, lr=1e-3),
    "freq_only": dict(epochs=60, batch_size=128, lr=1e-3),
    "stage1": dict(epochs=4, batch_size=32, lr=1e-4),
    "full_hybrid": dict(epochs=4, batch_size=32, lr=1e-4),
}
HF_CHECKPOINT_REPO = "sahilSpit/checkpoints"
HF_CHECKPOINT_PRIVATE = True

# Mock the functions that are not available
class MockApi:
    def create_repo(self, *args, **kwargs):
        pass
    def upload_file(self, *args, **kwargs):
        pass

class MockHfApi:
    def __init__(self, token):
        pass

class MockUserSecretsClient:
    def get_secret(self, key):
        return "dummy"

# We need to mock the modules that are not available
sys.modules['huggingface_hub'] = type('MockHuggingfaceHub', (), {'HfApi': MockHfApi})()
sys.modules['kaggle_secrets'] = type('MockKaggleSecrets', (), {'UserSecretsClient': MockUserSecretsClient})()
sys.modules['src.eval.cache_meta'] = type('MockCacheMeta', (), {
    'load_cache_meta': lambda x: {'branches': [], 'max_train_images': 0, 'semantic_model': '', 'semantic_dim': 0, 'preprocess': '', 'augment': False},
    'normalise_branches': lambda x: []
})()
sys.modules['src.eval.splits'] = type('MockSplits', (), {'load_split_spec': lambda x: None})()
sys.modules['src.eval.tables'] = type('MockTables', (), {'build_eval_tables': lambda x, y, z: {'at_0.5': {'overall': {}}}})()
sys.modules['src.eval.cache_meta'] = type('MockCacheMeta', (), {
    'load_cache_meta': lambda x: {'branches': [], 'max_train_images': 0, 'semantic_model': '', 'semantic_dim': 0, 'preprocess': '', 'augment': False},
    'normalise_branches': lambda x: []
})()

# Read the notebook
notebook_path = "notebooks/kaggle_train.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# We'll execute the cells in order
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    source = cell.source
    print(f"Executing cell {i}...")
    try:
        # Execute the source
        exec(source, globals())
    except Exception as e:
        print(f"Cell {i} failed with error: {e}")
        if i == 8:
            print("This is the training cell (cell 8)!")
        break
else:
    print("All cells executed successfully")