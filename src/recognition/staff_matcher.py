from __future__ import annotations


class StaffMatcher:
    """Utility for ranking gallery identities by similarity in a simple prototype workflow."""

    def __init__(self, threshold=0.30):
        self.threshold = threshold

    def match(self, query_embedding, gallery):
        if not gallery:
            return {"identity": "Unknown", "similarity": None, "status": "no_candidates", "candidates": []}

        ranked = []
        for staff_name, embeddings in gallery.items():
            if not embeddings:
                continue
            similarities = [float(similarity) for similarity in embeddings]
            best_similarity = max(similarities, default=0.0)
            ranked.append({"name": staff_name, "similarity": best_similarity})

        ranked.sort(key=lambda result: result["similarity"], reverse=True)
        if not ranked:
            return {"identity": "Unknown", "similarity": None, "status": "no_candidates", "candidates": []}

        best = ranked[0]
        if best["similarity"] < self.threshold:
            return {"identity": "Unknown", "similarity": best["similarity"], "status": "below_threshold", "candidates": ranked}
        return {"identity": best["name"], "similarity": best["similarity"], "status": "candidate_match", "candidates": ranked}
