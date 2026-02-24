# merge_embeddings.py
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    output_dir = Path("embeddings")
    
    # بارگذاری Text Embeddings
    text_file = output_dir / "snomed_text_embeddings.json"
    if not text_file.exists():
        raise FileNotFoundError(f"{text_file} not found.")
    with open(text_file, 'r') as f:
        text_embeddings = json.load(f)
    text_dim = len(next(iter(text_embeddings.values())))
    logger.info(f"Loaded {len(text_embeddings)} text embeddings (dim={text_dim})")

    # بارگذاری Graph Embeddings
    graph_file = output_dir / "snomed_structural_embeddings.json"
    if not graph_file.exists():
        raise FileNotFoundError(f"{graph_file} not found.")
    with open(graph_file, 'r') as f:
        graph_embeddings = json.load(f)
    graph_dim = len(next(iter(graph_embeddings.values())))
    logger.info(f"Loaded {len(graph_embeddings)} graph embeddings (dim={graph_dim})")

    # استخراج تمام SNOMED codeها
    all_codes = set(text_embeddings.keys()).union(graph_embeddings.keys())
    logger.info(f"Found {len(all_codes)} unique SNOMED codes for merging")

    # ادغام embeddings
    merged_embeddings = {}
    for code in all_codes:
        text_vec = np.array(text_embeddings.get(code, np.zeros(text_dim)))
        graph_vec = np.array(graph_embeddings.get(code, np.zeros(graph_dim)))
        merged_embeddings[code] = np.concatenate([text_vec, graph_vec]).tolist()

    # ذخیره merged
    merged_json_file = output_dir / "snomed_merged_embeddings.json"
    merged_npy_file = output_dir / "snomed_merged_embeddings.npy"

    with open(merged_json_file, 'w') as f:
        json.dump(merged_embeddings, f, indent=2)
    np.save(merged_npy_file, np.array(list(merged_embeddings.values())))

    logger.info(f"✅ Merged embeddings saved to {merged_json_file} and {merged_npy_file}")
    logger.info(f"Final embedding dimension: {text_dim + graph_dim}")
