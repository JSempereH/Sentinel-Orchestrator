"""Command-line entry points for reproducible analysis requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .workflow import AnalysisRequest, AnalysisWorkflow, estimate_request


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
    run = subparsers.add_parser("run", help="discover, download, process and fuse, then write the result cubes")
    run.add_argument("request")
    run.add_argument("output")
    run.add_argument("--max-workers", type=int, default=1, help="parallel product downloads/reads (each can need GBs of RAM)")
    run.add_argument("--gpt", default="gpt", help="SNAP GPT executable for sentinel1_backend='snap'")
    check = subparsers.add_parser("check-credentials", help="authenticate against every configured data provider")
    check.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    if args.command == "check-credentials":
        from .credentials import check_credentials

        results = check_credentials(args.env_file)
        print(json.dumps({service: result.to_dict() for service, result in results.items()}, indent=2))
        # Non-zero when a configured credential fails, so cron/CI can alert on it.
        return 1 if any(result.failed for result in results.values()) else 0
    request = AnalysisRequest.from_json(args.request)
    workflow = AnalysisWorkflow(request)
    if args.command == "plan":
        print(json.dumps({"steps": [step.__dict__ for step in workflow.plan.steps], "estimate": estimate_request(request).to_dict()}, indent=2))
    elif args.command == "discover":
        discovered: dict[str, object] = {sensor: [getattr(item, "name", getattr(item, "item_id", str(item))) for item in products] for sensor, products in workflow.discover().items()}
        discovered["auxiliary"] = {name: spec.to_dict() for name, spec in workflow.discover_auxiliary().items()}
        print(json.dumps(discovered, indent=2))
    elif args.command == "run":
        def progress(sensor: str, done: int, total: int) -> None:
            print(f"{sensor}: {done}/{total}", file=sys.stderr)

        output = Path(args.output)
        result = workflow.execute(output / "work", max_workers=args.max_workers, gpt=args.gpt, progress=progress)
        result_dir = result.save(output / "result")
        print(json.dumps({"result": str(result_dir), "variables": list(result.cube.data_vars), "provenance": result.provenance}, indent=2, default=str))
    else:
        datasets = workflow.acquire_auxiliary(args.output)
        print(json.dumps({name: {"variables": list(dataset.data_vars), "path": dataset.attrs.get("auxiliary_artifact_path")} for name, dataset in datasets.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
