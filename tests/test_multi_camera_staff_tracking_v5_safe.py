import importlib.util
import queue
import sys
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "multi_camera_staff_tracking_v5_safe.py"
)

spec = importlib.util.spec_from_file_location("staff_tracking_safe", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class SafeIdentityTests(unittest.TestCase):
    def setUp(self):
        mod.registry = mod.SharedRegistry()
        mod.event_queue = queue.Queue()
        mod.FACE_MATCH_THRESHOLD = 0.30
        mod.FACE_VOTES_REQUIRED = 3
        mod.FACE_VOTE_WINDOW_SEC = 20.0

    def make_track(self, camera="CAM-202", local_id=1, t=100.0):
        gp = mod.registry._new_global(t)
        lt = mod.LocalTrack(camera, local_id, gp.global_id, t)
        mod.registry.locals[(camera, local_id)] = lt
        return gp, lt

    def test_same_name_does_not_merge_global_ids(self):
        gp1, lt1 = self.make_track()
        gp2, lt2 = self.make_track("CAM-302", 2, 101.0)

        gp1.staff_name = "Prathamesh"
        gp2.staff_name = "Prathamesh"

        original_id = lt2.global_id
        mod.registry.link_confirmed_face(
            lt2, "Prathamesh", 102.0
        )

        self.assertEqual(lt2.global_id, original_id)
        self.assertNotEqual(lt1.global_id, lt2.global_id)

    def test_conflicting_names_block_attendance(self):
        gp, lt = self.make_track()

        for t in (100.0, 101.0, 102.0):
            mod.registry.update_face(
                lt, "Prathamesh", 0.8, t
            )

        self.assertEqual(
            mod.registry.get_name(lt), "Prathamesh"
        )

        mod.registry.update_face(
            lt, "Saransh", 0.8, 103.0
        )

        self.assertTrue(gp.identity_conflicted)
        mod.registry.maybe_attendance(lt, lt.camera_id)
        self.assertTrue(mod.event_queue.empty())

    def test_repeated_votes_confirm_one_name(self):
        gp, lt = self.make_track()

        for t in (100.0, 101.0, 102.0):
            mod.registry.update_face(
                lt, "Dhruval", 0.8, t
            )

        self.assertFalse(gp.identity_conflicted)
        self.assertEqual(
            mod.registry.get_name(lt), "Dhruval"
        )
    def test_expired_conflicting_votes_do_not_stay_active(self):
        gp, lt = self.make_track()

        for t in (100.0, 101.0, 102.0):
            mod.registry.update_face(
                lt, "Prathamesh", 0.8, t
            )

        mod.registry.update_face(
            lt, "Saransh", 0.8, 103.0
        )

        self.assertTrue(gp.identity_conflicted)

        # Both names' votes should be outside the 20-second window.
        mod.registry.update_face(
            lt, "Prathamesh", 0.8, 130.0
        )

        self.assertNotIn(
            "Saransh", [
                name
                for name, votes in gp.face_vote_times.items()
                if votes
            ]
        )    


if __name__ == "__main__":
    unittest.main()