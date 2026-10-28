from __future__ import annotations

import numpy as np


class BodyReID:
    """Small embedding-based body re-identification utility used for prototypes."""

    def __init__(self, dim=None):
        self.dim = dim

    @staticmethod
    def _as_vector(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Embedding is empty.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains non-finite values.")

        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Embedding norm is zero.")
        return vector / norm

    def extract_embedding(self, image, bbox=None):
        if image is None or np.asarray(image).size == 0:
            raise ValueError("Input image is empty or invalid.")

        arr = np.asarray(image, dtype=np.float32)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            arr = arr[y1:y2, x1:x2]
            if arr.size == 0:
                raise ValueError("Body crop is empty.")

        vector = arr.reshape(-1)
        if self.dim is not None and vector.size != self.dim:
            vector = vector[: self.dim]

        return self._as_vector(vector)

    def cosine_similarity(self, a, b):
        a_vec = self._as_vector(a)
        b_vec = self._as_vector(b)
        return float(np.dot(a_vec, b_vec))

    def match(self, query_embedding, gallery, threshold=0.5):
        if not gallery:
            return {"identity": "Unknown", "similarity": None, "status": "no_candidates", "candidates": []}

        ranked = []
        for name, embeddings in gallery.items():
            if not embeddings:
                continue
            sims = [self.cosine_similarity(query_embedding, embedding) for embedding in embeddings]
            ranked.append({"name": name, "similarity": max(sims)})

        ranked.sort(key=lambda item: item["similarity"], reverse=True)
        best = ranked[0]
        if best["similarity"] < threshold:
            return {"identity": "Unknown", "similarity": best["similarity"], "status": "below_threshold", "candidates": ranked}
        return {"identity": best["name"], "similarity": best["similarity"], "status": "candidate_match", "candidates": ranked}
