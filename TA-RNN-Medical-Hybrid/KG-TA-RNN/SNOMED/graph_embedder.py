# structural_embedder_flexible.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union
import networkx as nx
import logging
import pickle
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GraphSAGEEmbedder:
    """ساخت امبدینگ ساختاری SNOMED با GraphSAGE (قابل دریافت فایل یا لیست کدها)"""

    def __init__(self, embedding_dim: int = 256, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.graph = nx.Graph()
        self.code_to_idx = {}
        self.idx_to_code = {}
        logger.info(f"Initialized GraphSAGE embedder (dim={embedding_dim}) on {self.device}")

    def load_snomed_relations(self, snomed_dir: str = "data/Terminology") -> List[Dict]:
        """بارگذاری روابط SNOMED"""
        files = list(Path(snomed_dir).glob("der2_*Snapshot_*.txt"))
        relations = []
        for file in files:
            try:
                df = pd.read_csv(file, sep='\t', dtype=str, low_memory=False)
            except Exception:
                continue
            if df is not None and "active" in df.columns:
                df = df[df["active"] == "1"]
                for _, row in df.iterrows():
                    source = row.get("sourceId") or row.get("referencedComponentId")
                    target = row.get("destinationId") or row.get("target")
                    rel_type = row.get("typeId") or "other"
                    if source and target:
                        relations.append({"source": source, "target": target, "type": rel_type})
        logger.info(f"Loaded {len(relations)} SNOMED relations")
        return relations

    def build_graph(self, snomed_codes: List[str], relations: List[Dict]):
        """ساخت گراف SNOMED"""
        for i, code in enumerate(snomed_codes):
            self.code_to_idx[code] = i
            self.idx_to_code[i] = code
            self.graph.add_node(i, code=code)

        for rel in tqdm(relations, desc="Adding SNOMED edges"):
            source, target, rel_type = rel["source"], rel["target"], rel["type"]
            if source in self.code_to_idx and target in self.code_to_idx:
                idx1, idx2 = self.code_to_idx[source], self.code_to_idx[target]
                if rel_type == '116680003':
                    weight = 1.0
                elif rel_type == '363698007':
                    weight = 0.8
                elif rel_type == '246075003':
                    weight = 0.7
                else:
                    weight = 0.5
                self.graph.add_edge(idx1, idx2, weight=weight, type=rel_type)

    def add_cooccurrence_edges(self, mapping_file="data/icdToSnomedMapping/numeric_to_snomed_quick_map.json", seq_file="data/mimic/mimic_output.seqs"):
        """اضافه کردن edge بر اساس هم‌وقوعی در MIMIC"""
        try:
            with open(mapping_file, 'r') as f:
                num_to_snomed = json.load(f)
            with open(seq_file, 'rb') as f:
                seqs = pickle.load(f)

            cooccurrence = {}
            for patient_seq in tqdm(seqs[:1000], desc="Calculating co-occurrence"):
                snomed_set = set()
                for visit in patient_seq:
                    for num_code in visit:
                        code = str(num_code)
                        if code in num_to_snomed and num_to_snomed[code] in self.code_to_idx:
                            snomed_set.add(num_to_snomed[code])
                snomed_list = list(snomed_set)
                for i in range(len(snomed_list)):
                    for j in range(i + 1, len(snomed_list)):
                        pair = tuple(sorted([snomed_list[i], snomed_list[j]]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

            threshold = 5
            for (c1, c2), count in cooccurrence.items():
                if count >= threshold:
                    idx1, idx2 = self.code_to_idx[c1], self.code_to_idx[c2]
                    weight = np.log1p(count) / 10
                    if self.graph.has_edge(idx1, idx2):
                        self.graph[idx1][idx2]['weight'] += weight
                    else:
                        self.graph.add_edge(idx1, idx2, weight=weight, type='cooccurrence')
            logger.info("Added co-occurrence edges")
        except Exception as e:
            logger.warning(f"Could not add co-occurrence edges: {e}")

    def train_graphsage(self, num_epochs=50, lr=0.01):
        num_nodes = self.graph.number_of_nodes()
        adj = nx.to_numpy_array(self.graph, nodelist=range(num_nodes))
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        features = F.normalize(torch.randn(num_nodes, self.embedding_dim).to(self.device), p=2, dim=1)

        class SAGELayer(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.linear = nn.Linear(in_dim * 2, out_dim)
            def forward(self, x, adj):
                neigh = torch.matmul(adj, x) / (adj.sum(1, keepdim=True) + 1e-8)
                combined = torch.cat([x, neigh], dim=1)
                return F.normalize(F.relu(self.linear(combined)), p=2, dim=1)

        sage1 = SAGELayer(self.embedding_dim, self.embedding_dim).to(self.device)
        sage2 = SAGELayer(self.embedding_dim, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(list(sage1.parameters()) + list(sage2.parameters()), lr=lr)

        for epoch in tqdm(range(num_epochs), desc="Training GraphSAGE"):
            h1 = sage1(features, adj_tensor)
            h2 = sage2(h1, adj_tensor)
            similarity = torch.matmul(h2, h2.T)
            target = (adj_tensor > 0).float()
            loss = F.mse_loss(similarity, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h1 = sage1(features, adj_tensor)
            final_embeddings = sage2(h1, adj_tensor)
        return final_embeddings.cpu().numpy()

    def create_structural_embeddings(self, mapping: Union[str, List[str]] = "data/icdToSnomedMapping/numeric_to_snomed_quick_map.json",
                                     output_dir="embeddings"):
        """ساخت امبدینگ ساختاری از مسیر فایل یا لیست SNOMED کدها"""
        # تعیین snomed_codes
        if isinstance(mapping, list):
            snomed_codes = mapping
        else:
            with open(mapping, 'r') as f:
                num_to_snomed = json.load(f)
            snomed_codes = list(set(num_to_snomed.values()))

        # ساخت گراف
        relations = self.load_snomed_relations()
        self.build_graph(snomed_codes, relations)
        self.add_cooccurrence_edges()  # فایل پیش‌فرض mapper

        embeddings = self.train_graphsage()

        # ذخیره
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        emb_dict = {self.idx_to_code[i]: embeddings[i].tolist() for i in range(len(snomed_codes))}
        with open(output_path / "snomed_structural_embeddings.json", 'w') as f:
            json.dump(emb_dict, f)
        np.save(output_path / "snomed_structural_embeddings.npy", embeddings)

        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_nodes': len(snomed_codes),
            'num_edges': self.graph.number_of_edges(),
            'graph_density': nx.density(self.graph)
        }
        with open(output_path / "structural_embeddings_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Created structural embeddings for {len(snomed_codes)} SNOMED codes")
        return emb_dict

    def run(self, mapping: Union[str, List[str]] = "data/icdToSnomedMapping/numeric_to_snomed_quick_map.json",
            output_dir="embeddings"):
        return self.create_structural_embeddings(mapping, output_dir)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    mapper_json = Path("icdToSnomedMapping/numeric_to_snomed_quick_map.json")
    if not mapper_json.exists():
        raise FileNotFoundError("Mapper numeric_to_snomed_quick_map.json not found in embeddings/")

    with open(mapper_json, 'r') as f:
        mapping = json.load(f)

    snomed_codes = list(set(mapping.values()))
    logger.info(f"Found {len(snomed_codes)} unique SNOMED codes from mapper")

    embedder = GraphSAGEEmbedder(embedding_dim=256)
    embeddings = embedder.create_structural_embeddings(snomed_codes, output_dir="embeddings")
    logger.info("✅ Structural embeddings created successfully.")
