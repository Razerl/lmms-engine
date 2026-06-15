import argparse
import math
import os
import random
import sys
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from transformers import AutoProcessor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from lmms_engine.utils.data_utils import DataUtilities

dataset_path = "data/oms_sft_v1_3.yaml"
processor_name = "/home/rzli/data/data_ssd_smb/models/QwenVL/Qwen3-VL-4B-Instruct"

# Global processor for workers
processor = None


def init_worker(proc_name):
    global processor
    try:
        processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor in worker: {e}")


def calculate_tokens(item):
    global processor
    if processor is None:
        return 0, 0

    # Text tokens
    text_tokens = 0
    texts = item.get("texts", [])
    if texts:
        full_text = ""
        for turn in texts:
            # Simple concatenation.
            # Ideally we would add special tokens like <|im_start|> etc. but without a working template, this is the safest approximation.
            # We add a newline between turns.
            u = turn.get("user", "")
            a = turn.get("assistant", "")
            full_text += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"

        try:
            # Use tokenizer directly
            text_tokens = len(processor.tokenizer(full_text)["input_ids"])
        except Exception:
            text_tokens = 0

    # Image tokens
    image_tokens = 0
    images = item.get("images", [])

    # Only process if there are images
    if images and len(images) > 0:
        image_processor = processor.image_processor
        patch_size = getattr(image_processor, "patch_size", 14)
        merge_size = getattr(image_processor, "merge_size", 2)

        # Helper to calc tokens for one image
        def calc_one_image(size):
            if not isinstance(size, list):
                return 0

            # Qwen2-VL/Qwen3-VL logic:
            # tokens = ceil(H / (patch_size * merge_size)) * ceil(W / (patch_size * merge_size))
            # Note: This is an approximation. Real logic might involve padding to multiple of patch_size*merge_size.

            if len(size) == 2:  # H, W
                h, w = size
                # Effective patch size after merge
                effective_patch = patch_size * merge_size
                h_t = math.ceil(h / effective_patch)
                w_t = math.ceil(w / effective_patch)
                return h_t * w_t
            return 0

        image_size = item.get("image_size")

        if image_size is not None:
            if isinstance(image_size, list):
                if len(image_size) > 0 and isinstance(image_size[0], list):
                    # List of lists (multiple images)
                    for size in image_size:
                        image_tokens += calc_one_image(size)
                else:
                    # Single image size [H, W] (assuming one image or shared size?)
                    # If we have multiple images but one size list, it might be ambiguous.
                    # But usually it matches the structure.
                    # If it's just [H, W], treat as one image.
                    # random_size = random.choice([[66*16, 116*16], [46*16, 82*16], [32*16, 58*16], [24*16, 44*16]])
                    # image_tokens += calc_one_image(random_size)
                    image_tokens += calc_one_image(image_size)
        else:
            # Fallback if image_size is missing but images exist
            # Use 600x900 for each image
            default_size = [600, 900]
            tokens_per_image = calc_one_image(default_size)
            image_tokens = tokens_per_image * len(images)

    return text_tokens, image_tokens


def transform_number(number):
    """Transform large numbers into human-readable format."""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.2f}K"
    else:
        return str(number)


def main():
    print(f"Loading dataset from: {dataset_path}")
    data_list, _ = DataUtilities.load_yaml(dataset_path)

    # Use multiprocessing
    num_workers = min(32, cpu_count())
    print(f"Using {num_workers} workers for token calculation...")

    total_text_tokens = 0
    total_image_tokens = 0

    with Pool(num_workers, initializer=init_worker, initargs=(processor_name,)) as p:
        results = list(tqdm(p.imap(calculate_tokens, data_list, chunksize=1000), total=len(data_list)))

    for t, i in results:
        total_text_tokens += t
        total_image_tokens += i

    print("-" * 30)
    print(f"Total Text Tokens: {transform_number(total_text_tokens)}")
    print(f"Total Image Tokens: {transform_number(total_image_tokens)}")
    print(f"Total Tokens: {transform_number(total_text_tokens + total_image_tokens)}")
    print("-" * 30)


if __name__ == "__main__":
    main()
