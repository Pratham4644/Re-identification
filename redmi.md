# Workplace Monitoring Plan

This document is a work plan only. It does not change the current detection, recognition, or tracking code.

The goal is to answer two questions for enrolled employees, using the person re-identification work that already exists:

1. How many times each employee sits in a chair, and for how long.
2. How many times each employee leaves the office during working hours, and for how long.

Re-identification answers **who**. Zone logic answers **what they did**. The two stay separate so a face match is never treated as a chair event, and a chair overlap is never treated as a name.

## What already exists

The current project can already do the identity half of this work.

| Piece | Where it lives | What it gives this plan |
| --- | --- | --- |
| Person detection | `src/detection/person_detector.py` | A person box on each frame. YOLO, COCO class person. |
| Face embedding and gallery match | `src/recognition/face_recognizer.py` | A staff name, or `Unknown`, from `data/staff_gallery/<name>/face/`. |
| Face quality gate | `src/recognition/face_quality.py` | Size, blur, and detector-confidence checks before a face is trusted. |
| Face-to-person link | `src/detection/face_person_associator.py` | Which person box owns which face box. Geometry only. |
| Live person track | `scripts/live_person_tracking.py`, `scripts/live_person_recognition.py` | A camera-local track id from Ultralytics ByteTrack. |
| Multi-camera staff link | `scripts/multi_camera_staff_tracking_v4_fixed.py` | A cautious global id. Face votes can attach a staff name. Body similarity can continue a track. Body similarity does not invent a name. |
| First-seen attendance row | same multi-camera script | One `first_seen` row per named staff member per day. |

The camera, tracking, and app packages under `src/` are still scaffolding. The runnable behavior is in `scripts/`.

What the current project does **not** do yet:

- It does not know where a chair is.
- It does not know where the office door is.
- It does not time a sitting session.
- It does not time an absence.
- `first_seen` only records that a named person appeared. It is not an arrival, a sit-down, or a departure.

Those missing pieces are zone and time rules on top of the existing identity stream. They do not replace the gallery, the face model, or the tracker.

## The two results

Both results are daily, per enrolled employee.

### 1. Chair use

For each employee and each working day:

- `sit_count`: how many separate times they sat down.
- `sit_duration`: total time spent seated.
- Optional detail: which chair, which camera, start time, end time of each sit.

A sit is one continuous stay in a chair zone. Standing up and sitting again is a new sit. Leaning over a desk for a moment is not a sit.

### 2. Time outside the office

For each employee and each working day, counting only the configured working window:

- `exit_count`: how many times they left.
- `away_duration`: total time outside before they came back, clipped to working hours.
- Optional detail: exit time, return time, exit camera, return camera.

Leaving after the workday ends is not counted. Coming back the next morning is not an “away” interval from the previous evening. A person who is outside when the workday starts is not given a fake exit at the start of the day. Their first counted exit is a real crossing of the door during the window.

## One shared pipeline

Every camera, desk or door, produces the same kind of observation. Chair rules and door rules both read that observation. They do not run their own face models.

```text
RTSP frame
    -> person box                         PersonDetector / YOLO track
    -> camera-local track id              ByteTrack
    -> face crop, only when due           FaceRecognizer + FaceQualityChecker
    -> face box joined to person box      FacePersonAssociator
    -> staff name or Unknown              gallery cosine match + vote
    -> same person across cameras         existing global id / optional body ReID
    -> observation                         who, where, box, time
         |                          |
         v                          v
    chair zones                 door line
    sit start / sit end         exit / return
         |                          |
         +------------+-------------+
                      v
              daily report per employee
```

An observation is one row of facts, not a business decision:

| Field | Meaning |
| --- | --- |
| `timestamp` | When the box was seen. |
| `camera_id` | Which camera. |
| `local_track_id` | Tracker id on that camera. |
| `global_id` | Same person if another camera already linked them. |
| `staff_name` | Gallery name after enough agreeing face votes, otherwise `Unknown`. |
| `bbox` | Person box in that frame. |
| `face_similarity` | Best gallery score when a face was checked. Empty when no face was checked. |

`scripts/multi_camera_staff_tracking_v4_fixed.py` already writes a close cousin of this row to `track_events.csv`, plus one `first_seen` attendance row. The new work keeps that identity row and adds two new event streams beside it. It does not overload `first_seen` to mean “sat down” or “left”.

### How a name gets attached

This is the existing re-identification rule, reused as-is in spirit:

1. Enroll each employee once. One folder per person under `data/staff_gallery/<Name>/face/`. Each enrollment photo must contain exactly one face. That is what `FaceRecognizer.load_gallery()` already expects.
2. On a live frame, detect people and keep ByteTrack ids.
3. Run face recognition on a track only every few seconds, and only when `FaceQualityChecker` accepts the crop. A blurry or tiny face does not vote.
4. `FacePersonAssociator` assigns that face to the person box that contains it. The name sticks to the person track, not to a floating face.
5. Require several agreeing votes inside a short window before `Unknown` becomes a staff name. The multi-camera experiment already does this with a vote count, a time window, a similarity floor, and a margin over the next candidate.
6. After the name is confirmed, later frames of the **same track** keep that name even when the face is turned away. A person seated with their back to the camera stays named because the track continues, not because the back of their head is recognized.
7. `Unknown` never opens a chair session or an absence session that is attributed to an employee. Those intervals can be stored as unlabeled occupancy for debugging. They do not enter the employee report.
8. Optional body ReID, only with a real person-ReID checkpoint in `V4_REID_CHECKPOINT`, may connect a track that lost its face to a track already named on another camera. A body score by itself still must not assign a staff name. That rule already exists and stays.

Identity is sticky for the life of the track. It is not re-decided from a chair polygon or a door line.

### How a name moves from a desk camera to a door camera

This is the part that makes the two features one system.

```text
Desk camera                         Door camera
person track T1                     person track T2
face votes -> "Asha"                face not visible yet
sits in chair zone                  crosses the exit line
        \                               /
         \---- same global id ---------/
                     |
                     v
        sit session and exit session
        both belong to Asha
```

The link is the global id from the multi-camera experiment:

- Same confirmed face name on two cameras can link those tracks when only one live identity has that name.
- A body embedding can keep two views of one person on one global id when the face is missing on the second camera.
- Two people standing in the same camera are never merged.
- If the link is uncertain, the door event stays `Unknown` until a face vote on the door camera confirms it. A wrong name on an exit is worse than a delayed name.

Until that cross-camera link is trusted on real footage, each feature can still run alone: chairs report from desk cameras, exits report from the door camera. The combined “Asha sat for 40 minutes, then left for 12 minutes” report waits until the same `global_id` is reliable.

## Cameras and zones

Draw zones once per camera, in that camera’s image. Store them as config next to `config/cameras.json` and `config/settings.yaml`. The current scripts do not read those files yet. The plan is to make the new rules read them, while the existing scripts keep their own constants until they are wired later.

Suggested cameras:

| Camera role | Looks at | Zones |
| --- | --- | --- |
| Desk camera | Rows of chairs | One polygon per chair. A chair has a stable id, such as `CHAIR-12`. |
| Door camera | The office doorway, from inside | One line across the door, with an inside side and an outside side. |
| Optional aisle camera | Path between desks and door | No business zone. Used only to keep the global id alive between the desk and the door. |

Working hours live in config, for example `09:30` to `18:30` in the office timezone, with the office weekdays. A lunch window can be marked as allowed absence so it is reported separately from other exits. The detector does not decide policy. Config does.

## Feature 1. Sitting in a chair

### What “sitting” means here

A person is seated when the lower part of their person box stays inside one chair polygon. Use the bottom-center of the box, not the whole box. The head and shoulders often hang outside a chair polygon while the person is sitting. The feet and hips are the useful point.

Standing beside a chair, walking past, or reaching over a desk fails this test because the bottom-center does not stay inside the polygon.

Hold the decision:

- Enter the zone and stay inside for a short confirm time, about 2 seconds, before `sit_start`. This drops people walking through the frame.
- Leave the zone, and wait for a grace time, about 15 to 30 seconds, before `sit_end`. This keeps one session when the tracker blinks or the person shifts in the chair.
- If they are still inside at the end of the video or the track dies inside the zone, close the session at the last seen time and mark it `ended_by_track_loss` so the duration is not left open forever.

Pose estimation can be added later to separate sitting from standing inside the same polygon. It is not required for the first version. The polygon plus dwell time is the rule.

### Flow

```text
for each desk-camera frame:
    observations = people with tracks and names
    for each chair polygon:
        candidates = observations whose bottom-center is inside this chair
        if more than one candidate:
            keep the one with the largest overlap
            mark the frame ambiguous if both overlaps are close
        if one candidate has stayed inside for the confirm time:
            if this chair has no open session:
                open sit session
                staff_name = candidate name, or Unknown
        if the open session's person has been outside for the grace time:
            close sit session
            duration = end - start
```

One chair has at most one open session. A second person standing behind the chair must not steal the session. If the frame is ambiguous, keep the current session and do not switch the name.

### Session record

| Field | Example |
| --- | --- |
| `staff_name` | `Asha` or `Unknown` |
| `chair_id` | `CHAIR-12` |
| `camera_id` | `CAM-DESK-1` |
| `global_id` | the re-id id |
| `local_track_id` | the desk-camera track |
| `start` | time `sit_start` fired |
| `end` | time `sit_end` fired |
| `duration_sec` | `end - start` |
| `end_reason` | `left_zone` or `track_lost` |

### Daily numbers

For employee reports, sum only sessions whose `staff_name` is an enrolled person:

- `sit_count` = number of closed sessions that day.
- `sit_duration` = sum of `duration_sec`.

`Unknown` sessions stay in the raw log so a missed face can be reviewed. They are excluded from the employee total.

### Where re-id is required, and where it is not

The chair polygon can count an occupied chair with no name at all. That proves the zone is drawn correctly.

The employee report needs the name from the gallery. The usual moment to capture the face is when the person walks up to the chair and is still facing a camera. After they sit and turn away, the track id carries the name. If the track is born already seated, with no usable face, the session stays `Unknown` until a later face vote on that same track upgrades it. Upgrading rewrites the open session’s name. It does not start a second sit.

## Feature 2. Leaving during working hours

### What “left the office” means here

A person left when their track crosses the door line from the inside side to the outside side. They returned when a later track of the same employee crosses from outside to inside.

A person who is visible near the door but does not cross the line has not left. A person who disappears from a desk camera has not left either. Desk disappearance is not an exit. Only the door line is an exit.

Direction comes from which side of the line the bottom-center was on in the previous frames, and which side it is on after the crossing. One frame on the line is not enough. Require the point to be clearly inside, then clearly outside, within a short window.

### Flow

```text
working window = configured hours for this weekday

for each door-camera frame:
    for each named or unknown track:
        side = inside or outside from bottom-center vs the door line
        if previous side was inside and current side is outside:
            emit exit
        if previous side was outside and current side is inside:
            emit return

when an exit is emitted:
    if staff_name is Unknown:
        store an unlabeled exit for review
        stop
    if the timestamp is outside the working window:
        store it as after_hours
        stop
    if this employee already has an open absence:
        keep the earlier exit
        stop
    open an absence: staff_name, exit_time, exit_camera, global_id

when a return is emitted for the same staff_name or same global_id:
    if an absence is open:
        close it
        raw_away = return_time - exit_time
        counted_away = overlap of that interval with the working window
    if no absence is open:
        this is an arrival, not a return from a counted exit
```

### Session record

| Field | Meaning |
| --- | --- |
| `staff_name` | Enrolled name. |
| `global_id` | Re-id id that ties the exit track to the return track. |
| `exit_time` | Door crossing out. |
| `return_time` | Door crossing in. Empty while they are still out. |
| `away_sec` | Time outside that falls inside working hours. |
| `exit_count contribution` | 1 per closed or still-open absence that started inside working hours. |
| `status` | `away`, `returned`, `after_hours`. |

If the workday ends while an absence is still open, close the counted duration at the end of working hours and mark `still_out`. The next morning’s entry is a new arrival, not the end of yesterday’s interval.

### Daily numbers

- `exit_count` = absences that started inside working hours.
- `away_duration` = sum of `away_sec` for those absences.

An allowed lunch window, if configured, is subtracted or reported in its own column. It is still measured. It is labeled so it is not mixed with other exits.

### Where re-id is required

The door camera must attach a name, or inherit a name from a global id that was named on the desk or aisle camera.

The exit and the return are often different local tracks. ByteTrack on the door camera starts a new id when the person comes back. The gallery name, or the global id already linked to that name, is what joins the exit to the return. Without that join, the system can count crossings and still cannot say how long **that employee** was gone.

Body ReID helps when the person leaves facing away and returns facing away. It may connect the two door tracks to a global id that already has a name. It still may not create the name.

## How the two features stay connected

```text
                    staff gallery
                         |
                    face votes
                         |
                    global id  <---- optional body ReID
                    /            \
                   /              \
           chair session      door absence
           sit_count          exit_count
           sit_duration       away_duration
                   \              /
                    \            /
                     daily sheet
                     one row per
                     employee per day
```

One employee row for a day:

| Employee | Sit count | Seated time | Exit count | Time away during work |
| --- | --- | --- | --- | --- |
| Asha | 4 | 5 h 10 m | 2 | 48 m |
| Rohan | 2 | 6 h 05 m | 0 | 0 m |

The sit numbers come only from chair sessions. The away numbers come only from door sessions. They share `staff_name` and `global_id`, so the report can also show the order of events: sat from 10:02 to 11:15, left at 11:16, returned at 11:40, sat from 11:41 to 13:05.

A person can be seated and then leave. They should not be seated and outside at the same time. When a door exit is confirmed, any open sit session for that same global id is closed at the exit time if the desk camera has already lost them. The door event is the authority for leaving. The chair event is the authority for sitting.

## Work sequence

Do this in order. Each step is usable before the later steps exist.

1. **Enroll staff.** Finish `data/staff_gallery`. Confirm enrollment with the existing face checks. People who are not enrolled stay `Unknown` and stay out of employee totals.
2. **Fix camera roles.** Decide which RTSP stream is a desk camera and which is the door. Record that in config. Reuse `CAMERA_202_RTSP` and `CAMERA_302_RTSP` only as examples of how streams are passed today. Real desk and door ids should be named for the room.
3. **Draw zones on still frames.** One polygon per chair. One directed line for the door. Save the image points with the camera id. Verify by eye before any timing rule runs.
4. **Prove chair occupancy without names.** Count how often a chair is occupied and for how long, using person boxes and tracks only. Compare a short labeled recording. Tune confirm time and grace time here.
5. **Attach names on the desk camera.** Feed the same tracks through the existing face gallery, quality check, face-to-person association, and vote rule. Employee `sit_count` and `sit_duration` start here.
6. **Prove the door line without names.** Count inside-to-outside and outside-to-inside crossings. Tune the line until walk-bys do not count.
7. **Attach names on the door.** Same face path. `exit_count` and `away_duration` start here, filtered by working hours.
8. **Join desk and door with the global id.** Use the multi-camera link so a sit and a later exit belong to the same employee even if the door frame never got a clear face. Keep the rule that body similarity does not assign a new name.
9. **Write the daily sheet** from closed sit sessions and closed absences. Keep the raw events. The sheet is a sum, not a second detector.

`src/camera/`, `src/tracking/`, and `src/app/` are the natural place for this wiring later. This plan does not fill them in yet. The identity behavior they should call is the behavior already demonstrated in `src/detection/`, `src/recognition/`, and `scripts/multi_camera_staff_tracking_v4_fixed.py`.

## Output to keep

Store three logs. Paths can follow the current `outputs/` layout when the work is built.

| Log | One row is |
| --- | --- |
| Observations | A tracked person on one camera at one time, with name if known. |
| Sit sessions | One sit, from confirm to leave. |
| Absences | One exit during the day, closed by a return or by the end of working hours. |

The daily sheet is computed from the two session logs. It is not hand-edited.

## Limits to keep visible

These limits are already true of the current prototype, and they apply directly to both new features.

- Face similarity thresholds in the scripts are experimental. A score is not proof of identity. Validate on footage from these cameras before anyone uses the sheet for attendance or discipline.
- Recognition fails when the face is small, blurred, in profile, occluded, or lit differently from enrollment. Those intervals stay `Unknown` rather than being assigned to the closest employee.
- The face-to-person associator matches boxes in one frame. It does not create a track. The tracker does.
- Cross-camera body matching is off unless `V4_REID_CHECKPOINT` points at a person-ReID OSNet checkpoint. ImageNet weights are not acceptable for this.
- Chair polygons and the door line are camera-specific. If a camera is moved, the zones are redrawn. The gallery does not need to be rebuilt for that.
- Tracker ids reset when a person is fully occluded or leaves the frame. The global id and the gallery name are what reconnect them. If both fail, the system starts a new unlabeled interval instead of guessing.
- This is biometric monitoring of employees. Run it only with the proper authorization, and keep gallery images, embeddings, and event logs out of git. `data/` is already ignored for that reason.

## Short version of the connection

Re-identification stays the identity service: gallery in, staff name and global id out.

Chair zones turn a named track into sit count and seated time.

The door line turns a named track into exit count and time away during working hours.

The daily sheet joins those sessions on the staff name. It does not run a third model.
