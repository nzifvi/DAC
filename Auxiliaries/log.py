import json
from datetime import datetime
from pathlib import Path

def logRun(record, path=Path("results") / "runs.jsonl"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **record,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
