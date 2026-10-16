from pathlib import Path

import cv2
import insightface
import numpy as np


class FaceRecognizer:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(320 , 320))

    @staticmethod
    def normalize_embedding(embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)

        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid face embedding.")

        return embedding / norm

    def extract_embedding(self, image_path):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError(f"No face detected: {image_path}")

        if len(faces) > 1:
            raise ValueError(
                f"Multiple faces detected in enrollment image: {image_path}"
            )

        return self.normalize_embedding(faces[0].normed_embedding)

    def load_gallery(self, gallery_path):
        gallery_path = Path(gallery_path)
        gallery = {}

        for staff_dir in sorted(gallery_path.iterdir()):
            if not staff_dir.is_dir():
                continue

            face_dir = staff_dir / "face"

            if not face_dir.is_dir():
                continue

            embeddings = []

            for image_path in sorted(face_dir.iterdir()):
                if image_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp"
                }:
                    continue

                try:
                    embedding = self.extract_embedding(image_path)
                    embeddings.append(embedding)
                    print(f"Enrolled image: {staff_dir.name}/{image_path.name}")

                except ValueError as exc:
                    print(f"Skipped image: {exc}")

            if embeddings:
                gallery[staff_dir.name] = embeddings

        if not gallery:
            raise RuntimeError("No valid staff embeddings were enrolled.")

        return gallery

    def rank_candidates(self, query_embedding, gallery):
        query_embedding = self.normalize_embedding(query_embedding)
        results = []

        for staff_name, reference_embeddings in gallery.items():
            # Use the highest reference-image similarity for this
            # initial prototype. We will evaluate this strategy later.
            similarities = [
                float(np.dot(query_embedding, reference))
                for reference in reference_embeddings
            ]

            results.append({
                "name": staff_name,
                "similarity": max(similarities),
            })

        return sorted(
            results,
            key=lambda result: result["similarity"],
            reverse=True,
        )

    def identify(self, query_embedding, gallery, threshold=0.30):
        """
        Demonstration only.
        The threshold has not been validated for production use.
        """
        ranked = self.rank_candidates(query_embedding, gallery)

        if not ranked:
            return {
                "identity": "Unknown",
                "similarity": None,
                "status": "no_candidates",
                "candidates": [],
            }

        best = ranked[0]

        if best["similarity"] < threshold:
            return {
                "identity": "Unknown",
                "similarity": best["similarity"],
                "status": "below_threshold",
                "candidates": ranked,
            }

        return {
            "identity": best["name"],
            "similarity": best["similarity"],
            "status": "candidate_match",
            "candidates": ranked,
        }