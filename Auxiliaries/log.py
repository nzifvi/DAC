import json
from datetime import datetime
def logRun(record, path=RESULTS / "runs.jsonl"):
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **record
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
