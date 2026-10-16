class FacePersonAssociator:
    """
    Associate detected faces with person bounding boxes.

    Initial implementation uses face-center containment.
    This is intended for testing, not production tracking.
    """

    def __init__(self, min_containment=0.80):
        self.min_containment = min_containment

    def associate(self, faces, people):
        associations = []

        for face_index, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = map(
                float,
                face.bbox,
            )

            face_area = max(
                0.0,
                (fx2 - fx1) * (fy2 - fy1),
            )

            if face_area <= 0:
                continue

            best_match = None
            best_overlap = 0.0

            for person_index, person in enumerate(people):
                px1, py1, px2, py2 = person["bbox"]

                ix1 = max(fx1, px1)
                iy1 = max(fy1, py1)
                ix2 = min(fx2, px2)
                iy2 = min(fy2, py2)

                intersection_area = max(
                    0.0,
                    ix2 - ix1,
                ) * max(
                    0.0,
                    iy2 - iy1,
                )

                containment = intersection_area / face_area

                if (
                    containment >= self.min_containment
                    and containment > best_overlap
                ):
                    best_overlap = containment
                    best_match = person_index

            associations.append({
                "face_index": face_index,
                "person_index": best_match,
                "containment": best_overlap,
            })

        return associations