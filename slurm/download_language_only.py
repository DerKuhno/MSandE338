"""
Language-only dataset download.

Equivalent to running src/prepare_data/download_datasets.py with all calls
commented out except the two marked "REQUIRED FOR LANGUAGE SETTING" and
"REQUIRED FOR LANGUAGE, ARITHMETIC, AND WMDP SETTING" in the original file.

Run from the repo root:
    python slurm/download_language_only.py
"""
import os
import sys

# Allow imports from src/utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orjson
from datasets import load_dataset
from tqdm import tqdm
from src.utils.paths import CACHE_DIR, DATASET_DIR

OUTPUT_DIR = os.path.join(DATASET_DIR, "fineweb")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_dataset(
    dataset_id,
    output_filename,
    subset_name=None,
    split="train",
    cache_dir=CACHE_DIR,
    batch_size=1_000_000,
    max_rows=None,
):
    print(f"Downloading subset '{subset_name}' from '{dataset_id}'...")
    if subset_name is None:
        ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)
    else:
        ds = load_dataset(
            dataset_id, name=subset_name, split=split, streaming=True, cache_dir=cache_dir
        )
    total = ds.num_rows if hasattr(ds, "num_rows") and ds.num_rows is not None else None

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    buffer = []
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=total, desc=f"Processing {subset_name}", mininterval=5):
            buffer.append(orjson.dumps(sample).decode("utf-8"))
            count += 1
            if count % batch_size == 0:
                f.write("\n".join(buffer) + "\n")
                print(f"Processed {count} lines so far...")
                buffer.clear()
            if max_rows is not None and count >= max_rows:
                break
        if buffer:
            f.write("\n".join(buffer) + "\n")
            print(f"Processed {count} lines in total.")
    print("Finished streaming dataset to:", output_path)


# ====== REQUIRED FOR LANGUAGE SETTING ======
download_dataset(
    dataset_id="HuggingFaceFW/fineweb-2",
    subset_name="kor_Hang",
    output_filename="fineweb2_kor.jsonl",
    max_rows=10_000_000,
)

# ====== REQUIRED FOR LANGUAGE, ARITHMETIC, AND WMDP SETTING ======
download_dataset(
    dataset_id="HuggingFaceFW/fineweb-edu",
    subset_name="sample-10BT",
    output_filename="fineweb_eng_sample-10BT.jsonl",
    max_rows=10_000_000,
)
