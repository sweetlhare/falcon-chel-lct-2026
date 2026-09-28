"""Fail-closed validation for human-reviewed, additive redaction ledgers.

This module validates supplied metadata only.  It never creates rectangles,
marks records reviewed, reads images, or changes the automatic redactor.
"""
import argparse
import json
import math
import re
from pathlib import Path


SHA256 = re.compile(r"[0-9a-f]{64}")


class LedgerValidationError(ValueError):
    pass


def _records(document, rectangles_key="rectangles"):
    if not isinstance(document, dict):
        raise LedgerValidationError("document must be a JSON object")
    if "images" in document:
        if not isinstance(document["images"], dict):
            raise LedgerValidationError("images must be an object")
        rows = [{"image_id": key, rectangles_key: value}
                for key, value in document["images"].items()]
    else:
        rows = document.get("records")
        if not isinstance(rows, list):
            raise LedgerValidationError("records must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("image_id"), str) or not row["image_id"]:
            raise LedgerValidationError("every record needs a nonempty image_id")
        if row["image_id"] in result:
            raise LedgerValidationError("duplicate image_id: " + row["image_id"])
        result[row["image_id"]] = row
    return result


def _bbox(value, label):
    if not isinstance(value, list) or len(value) != 4:
        raise LedgerValidationError(label + " must be [x,y,w,h]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise LedgerValidationError(label + " must contain four finite numbers")
    x, y, w, h = value
    if w <= 0 or h <= 0:
        raise LedgerValidationError(label + " must have positive width and height")
    return tuple(value)


def _rectangles(value, bbox, label):
    if not isinstance(value, list):
        raise LedgerValidationError(label + " must be a list")
    x, y, w, h = bbox
    result = []
    for index, rectangle in enumerate(value):
        name = f"{label}[{index}]"
        if not isinstance(rectangle, list) or len(rectangle) != 4:
            raise LedgerValidationError(name + " must be [x1,y1,x2,y2]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rectangle):
            raise LedgerValidationError(name + " must contain four finite numbers")
        a, b, c, d = rectangle
        if not (a < c and b < d):
            raise LedgerValidationError(name + " must have positive area")
        if a < x or b < y or c > x + w or d > y + h:
            raise LedgerValidationError(name + " lies outside the exact bbox")
        result.append(tuple(rectangle))
    return result


def _covered(rectangle, union):
    """Exact axis-aligned coverage test using coordinate-compressed cells."""
    a, b, c, d = rectangle
    clipped = [(max(a, x1), max(b, y1), min(c, x2), min(d, y2))
               for x1, y1, x2, y2 in union
               if max(a, x1) < min(c, x2) and max(b, y1) < min(d, y2)]
    xs = sorted({a, c, *(v for r in clipped for v in (r[0], r[2]))})
    ys = sorted({b, d, *(v for r in clipped for v in (r[1], r[3]))})
    for left, right in zip(xs, xs[1:]):
        for top, bottom in zip(ys, ys[1:]):
            px, py = (left + right) / 2, (top + bottom) / 2
            if not any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in clipped):
                return False
    return True


def audit_ledger(expected_document, baseline_document, ledger_document):
    """Return a complete audit report; ``valid`` is the fail-closed admission bit."""
    errors = []
    try:
        expected = _records(expected_document)
        baseline = _records(baseline_document)
        ledger = _records(ledger_document)
    except LedgerValidationError as error:
        return {"valid": False, "errors": [str(error)], "expected": 0, "reviewed": 0}

    expected_ids, baseline_ids, ledger_ids = set(expected), set(baseline), set(ledger)
    for label, ids in (("baseline", baseline_ids), ("ledger", ledger_ids)):
        missing = sorted(expected_ids - ids)
        extra = sorted(ids - expected_ids)
        if missing:
            errors.append(f"{label} missing IDs: {missing}")
        if extra:
            errors.append(f"{label} extra IDs: {extra}")
    if ledger_document.get("reviewed") is not True:
        errors.append("top-level reviewed must be true after real review")

    checked = 0
    for image_id in sorted(expected_ids & baseline_ids & ledger_ids):
        try:
            source = expected[image_id]
            candidate = ledger[image_id]
            digest = source.get("source_sha256")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise LedgerValidationError("expected source_sha256 must be lowercase SHA-256")
            if candidate.get("source_sha256") != digest:
                raise LedgerValidationError("source_sha256 mismatch")
            bbox = _bbox(source.get("bbox"), "expected bbox")
            if _bbox(candidate.get("bbox"), "ledger bbox") != bbox:
                raise LedgerValidationError("bbox mismatch")
            if candidate.get("status") != "reviewed":
                raise LedgerValidationError("status must be reviewed")
            if not isinstance(candidate.get("reviewer"), str) or not candidate["reviewer"].strip():
                raise LedgerValidationError("reviewer must be nonempty")
            old_value = baseline[image_id].get("rectangles", baseline[image_id].get("redactions"))
            old = _rectangles(old_value, bbox, "baseline rectangles")
            new = _rectangles(candidate.get("rectangles"), bbox, "ledger rectangles")
            for index, rectangle in enumerate(old):
                if not _covered(rectangle, new):
                    raise LedgerValidationError(f"baseline rectangle {index} is not fully covered")
            checked += 1
        except LedgerValidationError as error:
            errors.append(f"{image_id}: {error}")
    return {"valid": not errors, "errors": errors, "expected": len(expected),
            "reviewed": checked, "monotone_records": checked if not errors else None}


def validate_ledger(expected_document, baseline_document, ledger_document):
    report = audit_ledger(expected_document, baseline_document, ledger_document)
    if not report["valid"]:
        raise LedgerValidationError("; ".join(report["errors"]))
    return report


def _load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "validate"))
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_ledger(_load(args.expected), _load(args.baseline), _load(args.ledger))
    print(json.dumps(report, indent=2))
    if args.command == "validate" and not report["valid"]:
        return 2
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
