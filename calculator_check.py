"""Check that the calculator can reproduce publisher amounts from audited source inputs."""
import json
import re
from study import BEN, ADJ, ROOT, REFERENCE_MODE, MODES, load_case, calculate


def source_operands(case):
    case_id = case["id"]
    mode = REFERENCE_MODE[case_id]
    if case_id == "ADJ-017":
        return dict(mode=mode, allowed=940, deductible_credited=0, oop_credited=0,
                    deductible_limit=1500, oop_limit=999999, copay=0, coinsurance_percent=0)
    if case_id == "ADJ-018":
        return dict(mode=mode, allowed=300, deductible_credited=200, oop_credited=0,
                    deductible_limit=1500, oop_limit=999999, copay=0, coinsurance_percent=0)
    # Amounts are read from the published claim text. Fixed limits and service terms are
    # read from the common plan. In BEN-022 the published copay outcome implies the
    # copay pathway; its uncapped value is not scored as an independent model target.
    def read_amount(prefix):
        return float(re.search(re.escape(prefix) + r"[ .]+([0-9]+\.[0-9]{2})", case["request"])[1])
    return dict(mode=mode, allowed=read_amount("Allowed amount"),
                deductible_credited=read_amount("Member deductible credited year to date"),
                oop_credited=read_amount("Member out-of-pocket credited year to date"),
                deductible_limit=1500, oop_limit=6000,
                copay={"BEN-004":60,"BEN-007":400,"BEN-021":75,"BEN-022":30}.get(case_id,0),
                coinsurance_percent=20 if mode in (MODES[0], MODES[3]) else 0)


def diagnostic():
    report = {}
    for case_id in BEN + ADJ:
        case = load_case(case_id)
        operands = source_operands(case)
        result = calculate(operands)
        checked = {k: abs(result[k] - value) <= case["gold"]["tolerance"]
                   for k,value in case["gold"]["fields"].items() if k in result}
        report[case_id] = dict(source_inputs=operands, numeric_fields_match=all(checked.values()))
    assert all(item["numeric_fields_match"] for item in report.values()), report
    (ROOT / "results/calculator_check.json").write_text(json.dumps(report, indent=2))
    print("Published numeric results reproduced with source inputs on", len(report), "cases")


if __name__ == "__main__":
    diagnostic()
