"""Command-line entry points for reproducible analysis requests."""

from __future__ import annotations

import argparse
import json

from .workflow import AnalysisRequest, AnalysisWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel-analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("request")
    discover = subparsers.add_parser("discover")
    discover.add_argument("request")
    auxiliary = subparsers.add_parser("auxiliary")
    auxiliary.add_argument("request")
    auxiliary.add_argument("output")
    args = parser.parse_args(argv)
    request = AnalysisRequest.from_json(args.request)
    workflow = AnalysisWorkflow(request)
    if args.command == "plan":
        print(json.dumps({"steps": [step.__dict__ for step in workflow.plan.steps]}, indent=2))
    elif args.command == "discover":
        discovered: dict[str, object] = {sensor: [getattr(item, "name", getattr(item, "item_id", str(item))) for item in products] for sensor, products in workflow.discover().items()}
        discovered["auxiliary"] = {name: spec.to_dict() for name, spec in workflow.discover_auxiliary().items()}
        print(json.dumps(discovered, indent=2))
    else:
        datasets = workflow.acquire_auxiliary(args.output)
        print(json.dumps({name: {"variables": list(dataset.data_vars), "path": dataset.attrs.get("auxiliary_artifact_path")} for name, dataset in datasets.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
