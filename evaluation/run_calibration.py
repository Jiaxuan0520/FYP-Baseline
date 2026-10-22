from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_eval import build_demo_sessions, build_controllers, evaluate_controller


def main() -> None:
    sessions = build_demo_sessions()
    results = {}
    for controller in build_controllers():
        metrics = evaluate_controller(controller, sessions)
        results[controller.name] = {
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "auroc": metrics["auroc"],
        }
    for name, metrics in results.items():
        print(name, metrics)


if __name__ == "__main__":
    main()
