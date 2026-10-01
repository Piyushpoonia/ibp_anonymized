from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


def configured_classes(protocol_path: Path) -> tuple[str, ...]:
    module = ast.parse(protocol_path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "OBJECT_CLASSES"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return tuple(str(item) for item in value)
    raise RuntimeError(f"OBJECT_CLASSES was not found in {protocol_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    configured = configured_classes(project / "ibp_model/protocol.py")
    observed: dict[str, dict[str, object]] = defaultdict(
        lambda: {"semantic_ids": set(), "sequences": set(), "observations": 0}
    )
    category_files = sorted(
        (project / "predicates_v3_canonical_candidate/parts").glob("*/categories.json")
    )
    if len(category_files) != 9:
        raise SystemExit(f"Expected 9 category files; found {len(category_files)}")

    for path in category_files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            label = str(row["raw_label"])
            observed[label]["semantic_ids"].add(int(row["semantic_id"]))
            observed[label]["sequences"].add(path.parent.name)
            observed[label]["observations"] += int(row["observation_count"])

    missing = sorted(set(observed) - set(configured))
    absent = sorted(set(configured) - set(observed))
    duplicate_ids = {
        label: sorted(values["semantic_ids"])
        for label, values in observed.items()
        if len(values["semantic_ids"]) != 1
    }
    report = {
        "schema_version": "IBP-K360-object-vocabulary-audit-v1.0.0",
        "configured_classes": list(configured),
        "configured_class_count": len(configured),
        "observed_class_count": len(observed),
        "missing_from_model": missing,
        "configured_but_absent": absent,
        "labels_with_multiple_semantic_ids": duplicate_ids,
        "passed": not missing and not absent and not duplicate_ids,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Object-vocabulary audit failed; no jobs were submitted.")


if __name__ == "__main__":
    main()
