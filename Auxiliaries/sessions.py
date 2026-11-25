import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

def findSessions(root=Path("dataset"), pattern=r"session_\d{8}_\d{6}"):
    regex = re.compile(pattern)
    sessions = sorted(folder for folder in Path(root).iterdir() if folder.is_dir() and regex.fullmatch(folder.name) and (folder / "frames.csv").exists())
    if not sessions:
        raise FileNotFoundError(f"No folders matching {pattern!r} with a frames.csv under {Path(root).resolve()}")
    return sessions


def listSessions(root=Path("dataset"), pattern=r"session_\d{8}_\d{6}"):
    sessions = findSessions(root, pattern)
    for index, folder in enumerate(sessions):
        print(f"[{index}] {folder.name}: {len(pd.read_csv(folder / 'frames.csv')):,} frames")
    return sessions


def selectSessions(root=Path("dataset"), pattern=r"session_\d{8}_\d{6}", order=None):
    sessions = findSessions(root, pattern)
    if order is None:
        return sessions
    order = list(order)
    if len(set(order)) != len(order):
        raise ValueError(f"Session order {order} uses a session more than once")
    invalid = [i for i in order if not 0 <= i < len(sessions)]
    if invalid:
        available = ", ".join(f"[{i}] {folder.name}" for i, folder in enumerate(sessions))
        raise IndexError(f"Session order {order} has no session at {invalid}. Available: {available}")
    return [sessions[i] for i in order]

def _place(source, target, copy):
    if not copy:
        try:
            os.link(source, target)
            return "hard-linked"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copied"


def mergeSessions(root=Path("dataset"), unified=Path("dataset_unified"), pattern=r"session_\d{8}_\d{6}", order=None, copy=False, overwrite=False):
    unified = Path(unified)
    sessions = selectSessions(root, pattern, order)
    if unified.exists():
        if not overwrite:
            raise FileExistsError(f"{unified} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(unified)
    (unified / "rgb").mkdir(parents=True)
    (unified / "depth").mkdir(parents=True)

    rows, summary, actions, frameId = [], [], set(), 0
    for session in sessions:
        table = pd.read_csv(session / "frames.csv")
        if "source_session" in table.columns:
            raise ValueError(f"{session} is already a unified session; merge the original recordings instead")
        absent = {"frame_id", "rgb", "depth"} - set(table.columns)
        if absent:
            raise ValueError(f"{session / 'frames.csv'} is missing columns: {', '.join(sorted(absent))}")
        if table["frame_id"].duplicated().any():
            raise ValueError(f"Duplicate frame ID in {session / 'frames.csv'}")

        first = frameId + 1
        for row in table.sort_values("frame_id").itertuples(index=False):
            rgbSource, depthSource = session / row.rgb, session / row.depth
            for path in (rgbSource, depthSource):
                if not path.exists():
                    raise FileNotFoundError(f"{path} (listed in {session / 'frames.csv'})")
            frameId += 1
            rgbName = f"rgb/{frameId:06d}{rgbSource.suffix}"
            depthName = f"depth/{frameId:06d}{depthSource.suffix}"
            actions.add(_place(rgbSource, unified / rgbName, copy))
            actions.add(_place(depthSource, unified / depthName, copy))
            rows.append({"frame_id": frameId, "rgb": rgbName, "depth": depthName,
                         "source_session": session.name, "source_frame_id": int(row.frame_id)})
        summary.append({"session": session.name, "frames": len(table),
                        "first_frame_id": first, "last_frame_id": frameId})
        print(f"{session.name}: {len(table):,} frames -> frame_id {first}-{frameId}")

    if not rows:
        raise FileNotFoundError("The selected sessions contain no frames")
    pd.DataFrame(rows).to_csv(unified / "frames.csv", index=False)
    with open(unified / "merge.json", "w") as f:
        json.dump({"pattern": pattern, "order": None if order is None else list(order),
                   "frames": frameId, "sessions": summary}, f, indent=2)
    print(f"Wrote {unified}: {frameId:,} frames from {len(sessions)} sessions ({' and '.join(sorted(actions))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge recording sessions into one unified session for the DAC.")
    parser.add_argument("--list", action="store_true", help="only list the matching sessions and their indices")
    parser.add_argument("--order", type=int, nargs="+", default=None, help="indices to merge, in order (default: every match, in name order)")
    parser.add_argument("--pattern", default=r"session_\d{8}_\d{6}", help="regex for session folder names")
    parser.add_argument("--root", type=Path, default=Path("dataset"), help="folder holding the session folders")
    parser.add_argument("--out", type=Path, default=Path("dataset_unified"), help="the unified session folder to create")
    parser.add_argument("--copy", action="store_true", help="copy files instead of hard-linking them")
    parser.add_argument("--overwrite", action="store_true", help="rebuild --out if it already exists")
    args = parser.parse_args()

    listSessions(args.root, args.pattern)
    if not args.list:
        print()
        mergeSessions(args.root, args.out, args.pattern, args.order, args.copy, args.overwrite)