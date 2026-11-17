"""Score only saved responses against the published references."""
import json
import math
from pathlib import Path
import statistics
from decimal import InvalidOperation
from study import ROOT, BEN, ADJ, PA, REFERENCE_MODE, calculate, load_case


def matches(actual, expected, tolerance):
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip().upper().removeprefix("SECTION ").strip() == expected.upper()
    return type(actual) in (float, int) and math.isfinite(actual) and abs(actual - expected) <= tolerance


def analyze():
    results = ROOT / "results"
    expected_sources = json.loads((results / "sources.json").read_text(encoding="utf-8"))
    reference = {}
    for source in expected_sources:
        case = load_case(source["id"])
        if case["input_sha256"] != source["input_sha256"]:
            raise ValueError(f"Published case inputs changed for {case['id']}")
        reference[case["id"]] = case["gold"]
    rows = [json.loads(line) for line in (results / "predictions.jsonl").read_text().splitlines()]
    assert {(r["id"], r["method"]) for r in rows} == (
        {(i, m) for i in BEN + ADJ for m in ("direct", "hybrid")} | {(i, "documentation") for i in PA}
    ) and len(rows) == 40
    details = []
    for row in rows:
        case_id, method = row["id"], row["method"]
        raw = row["prediction"] or {}
        gold = reference[case_id]
        item = dict(id=case_id, method=method, finish=row["finish"], seconds=row["seconds"], valid=bool(raw))
        if case_id in PA:
            expected_criteria = set(gold["citations"])
            criteria = set(c.upper().strip() for c in raw.get("criteria", []))
            overlap = len(criteria & expected_criteria)
            item.update(decision_ok=raw.get("decision") == gold["label"],
                        expected_decision=gold["label"], decision=raw.get("decision"),
                        criteria_f1=2 * overlap / (len(criteria) + len(expected_criteria))
                        if criteria or expected_criteria else float(bool(raw)),
                        criteria_ok=bool(raw) and criteria == expected_criteria,
                        expected_criteria=sorted(expected_criteria), criteria=sorted(criteria))
        else:
            outcome = raw.copy()
            if method == "hybrid" and raw:
                try:
                    outcome.update(calculate(raw))
                except (ValueError, InvalidOperation, OverflowError, TypeError):
                    item["valid"] = False
            fields = gold["fields"]
            checks = {key: matches(outcome.get(key), value, gold["tolerance"])
                      for key, value in fields.items()}
            member = outcome.get("member_responsibility")
            item.update(mode_ok=raw.get("mode") == REFERENCE_MODE[case_id],
                        mode=raw.get("mode"), expected_mode=REFERENCE_MODE[case_id],
                        member=member, expected_member=fields["member_responsibility"],
                        amount_ok=checks["member_responsibility"] and checks["plan_paid"],
                        all_fields_ok=all(checks.values()), field_checks=checks,
                        member_error=abs(member - fields["member_responsibility"])
                        if type(member) in (float, int) and math.isfinite(member) else None)
        details.append(item)
    summary = {}
    for method in ("direct", "hybrid", "documentation"):
        group = [item for item in details if item["method"] == method]
        summary[method] = metrics(group)
    summary["payment_pairs"] = {
        "hybrid_only_amount": sum(h["amount_ok"] and not d["amount_ok"] for d, h in pairs(details)),
        "direct_only_amount": sum(d["amount_ok"] and not h["amount_ok"] for d, h in pairs(details)),
        "both_amount": sum(d["amount_ok"] and h["amount_ok"] for d, h in pairs(details)),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results / "case_scores.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def pairs(details):
    payment = {(x["id"], x["method"]): x for x in details if x["id"] not in PA}
    return [(payment[i, "direct"], payment[i, "hybrid"]) for i in BEN + ADJ]


def metrics(group):
    counts = {"n": len(group), "valid": sum(item["valid"] for item in group),
              "median_seconds": statistics.median(item["seconds"] for item in group)}
    if group[0]["method"] == "documentation":
        pend = [item for item in group if item["expected_decision"] == "pend"]
        counts.update(decision=sum(x["decision_ok"] for x in group),
                      decision_and_criteria=sum(x["decision_ok"] and x["criteria_ok"] for x in group),
                      mean_criteria_f1=statistics.mean(x["criteria_f1"] for x in group),
                      pend_n=len(pend), pend_correct=sum(x["decision_ok"] for x in pend),
                      false_pend=sum(x["decision"] == "pend" and x["expected_decision"] != "pend" for x in group))
    else:
        conflicts = [item for item in group if item["id"] in ADJ]
        errors = [item["member_error"] for item in group if item["member_error"] is not None]
        counts.update(mode=sum(x["mode_ok"] for x in group), amount=sum(x["amount_ok"] for x in group),
                      all_fields=sum(x["all_fields_ok"] for x in group),
                      mode_and_amount=sum(x["mode_ok"] and x["amount_ok"] for x in group),
                      right_mode_wrong_amount=sum(x["mode_ok"] and not x["amount_ok"] for x in group),
                      wrong_mode_right_amount=sum(not x["mode_ok"] and x["amount_ok"] for x in group),
                      governing_document=sum(x["field_checks"]["governing_document"] for x in conflicts),
                      governing_section=sum(x["field_checks"]["governing_section"] for x in conflicts),
                      numeric_n=len(errors), member_mae=statistics.mean(errors) if errors else None)
    return counts


if __name__ == "__main__":
    analyze()
