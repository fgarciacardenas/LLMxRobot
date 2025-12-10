# inference/rag_offline.py
import os, pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import faiss  # pip install faiss-cpu (optional on aarch64)
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

@dataclass
class LocalEmbeddings:
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: Optional[str] = None  # "cpu"/"cuda"/None

    def __post_init__(self):
        # Lazy import to avoid cost if --rag is not used
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, batch_size=64, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode([text], batch_size=1, normalize_embeddings=True)[0]
        return vec.tolist()

class OfflineRetriever:
    """
    Offline embeddings + ANN. Persists 2 files:
      - {index_path}.meta.pkl  (ids + texts)
      - {index_path}.faiss     (or {index_path}.npz if FAISS not available)
    """
    def __init__(self, index_path: str, embeddings: Optional[LocalEmbeddings] = None):
        self.index_path = index_path
        self.emb = embeddings or LocalEmbeddings()
        self._texts: List[str] = []
        self._ids: List[str] = []
        self._dim: Optional[int] = None
        self._index = None

    # ---------- build ----------
    def build(self, texts: List[str], ids: Optional[List[str]] = None):
        ids = ids or [str(i) for i in range(len(texts))]
        vecs = np.asarray(self.emb.embed_documents(texts), dtype="float32")
        self._dim = vecs.shape[1]
        self._texts, self._ids = texts, ids

        if _HAS_FAISS:
            index = faiss.IndexHNSWFlat(self._dim, 32)
            index.hnsw.efConstruction = 200
            index.add(vecs)
            self._index = index
        else:
            # NumPy fallback (linear scan)
            self._index = vecs

        self._persist()

    def _persist(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path + ".meta.pkl", "wb") as f:
            pickle.dump({"texts": self._texts, "ids": self._ids}, f)
        if _HAS_FAISS:
            faiss.write_index(self._index, self.index_path + ".faiss")
        else:
            np.savez_compressed(self.index_path + ".npz", vecs=self._index)

    # ---------- load ----------
    def load(self):
        with open(self.index_path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._texts, self._ids = meta["texts"], meta["ids"]

        faiss_path = self.index_path + ".faiss"
        npz_path = self.index_path + ".npz"
        if _HAS_FAISS and os.path.exists(faiss_path):
            self._index = faiss.read_index(faiss_path)
            self._dim = self._index.d
        else:
            arr = np.load(npz_path)
            self._index = arr["vecs"]
            self._dim = self._index.shape[1]

    # ---------- query ----------
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        q = np.asarray(self.emb.embed_query(query), dtype="float32").reshape(1, -1)
        if _HAS_FAISS:
            D, I = self._index.search(q, k)
            # unit vectors -> cos = 1 - D/2
            sims = 1.0 - (D[0] / 2.0)
            return [(self._ids[i], self._texts[i], float(sims[j])) for j, i in enumerate(I[0])]
        else:
            sims = (q @ self._index.T)[0]  # cosine, vectors normalized
            idx = np.argsort(-sims)[:k]
            return [(self._ids[i], self._texts[i], float(sims[i])) for i in idx]
