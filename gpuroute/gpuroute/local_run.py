"""Run one job on this machine. Invoked by gpuroute as a detached subprocess."""
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "templates"))
import train_core  # noqa: E402


def main():
    spec = json.loads(sys.argv[1])
    deadline = time.time() + spec["max_seconds"]

    # Hard stop that holds even if the job hangs outside the training loop (e.g. a
    # stalled download) and even if nobody is polling.
    def watchdog():
        train_core.emit("stopped_at_budget", reason="time cap reached")
        os._exit(3)
    timer = threading.Timer(spec["max_seconds"] + 20, watchdog)
    timer.daemon = True
    timer.start()
    try:
        train_core.run(spec["device"], spec["model"], spec["dataset"], spec["examples"], spec["seq_len"],
                       spec["epochs"], out_dir=spec["out_dir"], deadline=deadline)
    except Exception as e:
        train_core.emit("error", error=f"{e.__class__.__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
