import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# -------------------------------
# Logger
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# TextEmbedder Class
# -------------------------------
class TextEmbedder:
    """ساخت امبدینگ متنی برای SNOMED با BioClinicalBERT"""

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Initialized BioClinicalBERT on {self.device}")

    def load_snomed_descriptions(self, snomed_dir: str = "data/Terminology") -> Dict[str, str]:
        """بارگذاری توضیحات SNOMED با جلوگیری از NaN / float"""
        desc_path = Path(snomed_dir) / "sct2_Description_Snapshot-en_INT_20251201.txt"
        if not desc_path.exists():
            logger.warning("SNOMED descriptions file not found")
            return {}

        import pandas as pd
        df = pd.read_csv(desc_path, sep='\t', dtype=str)
        df = df[df['active'] == '1']

        descriptions = {}
        for _, row in df.iterrows():
            code = str(row['conceptId'])
            term = row['term']
            # اگر term خالی یا NaN بود ردش کن
            if not isinstance(term, str) or not term.strip():
                continue
            descriptions.setdefault(code, []).append(term.strip())

        # انتخاب طولانی‌ترین توضیح برای هر کد
        final_desc = {code: max(terms, key=len) for code, terms in descriptions.items() if terms}
        logger.info(f"Loaded {len(final_desc)} SNOMED descriptions")
        return final_desc

    def embed_batch(self, texts: List[str], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt",
                                    truncation=True, max_length=max_length,
                                    padding=True)
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    def create_text_embeddings(self, snomed_codes: List[str], descriptions: Dict[str,str],
                               output_dir: str = "GraphOutput") -> Dict[str, list]:
        texts, valid_codes = [], []
        for code in snomed_codes:
            texts.append(descriptions.get(code, "Medical concept"))
            valid_codes.append(code)

        embeddings = self.embed_batch(texts)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        emb_dict = {code: embeddings[i].tolist() for i, code in enumerate(valid_codes)}

        # ذخیره JSON و numpy
        with open(output_path / "snomed_text_embeddings.json", 'w') as f:
            json.dump(emb_dict, f, indent=2)
        np.save(output_path / "snomed_text_embeddings.npy", embeddings)

        # ذخیره metadata
        meta = {'model':'BioClinicalBERT','embedding_dim':self.embedding_dim,'num_codes':len(valid_codes)}
        with open(output_path / "text_embeddings_metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✓ Created text embeddings for {len(valid_codes)} SNOMED codes")
        return emb_dict

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    mapper_json = Path("icdToSnomedMapping/numeric_to_snomed_quick_map.json")
    if not mapper_json.exists():
        raise FileNotFoundError("Mapper numeric_to_snomed_quick_map.json not found in icdToSnomedMapping/")

    with open(mapper_json, 'r') as f:
        mapping = json.load(f)

    snomed_codes = list(set(mapping.values()))
    logger.info(f"Found {len(snomed_codes)} unique SNOMED codes from mapper")

    embedder = TextEmbedder()
    descriptions = embedder.load_snomed_descriptions("data/Terminology")
    embeddings = embedder.create_text_embeddings(snomed_codes, descriptions, output_dir="GraphOutput")
