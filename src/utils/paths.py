import os

# Override any of these with environment variables to avoid editing source.
# Set WORKSPACE_DIR to redirect all paths away from the default /workspace prefix.
_WORKSPACE = os.environ.get('WORKSPACE_DIR', '/workspace')
CACHE_DIR     = os.environ.get('CACHE_DIR',      f'{_WORKSPACE}/.cache')
DATASET_DIR   = os.environ.get('DATASET_DIR',    f'{_WORKSPACE}/datasets')
MODEL_DIR     = os.environ.get('MODEL_DIR',      f'{_WORKSPACE}/models/non-wmdp')
WMDP_MODEL_DIR= os.environ.get('WMDP_MODEL_DIR', f'{_WORKSPACE}/models/wmdp')
WANDB_API_KEY_PATH = "tokens/wandb_token.txt"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = CACHE_DIR
