"""Evaluate Parity 1.0 cases as published, using a loopback llama.cpp server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = "https://paritybench.ai"
BEN = [f"BEN-{n:03}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 21, 22)]
ADJ = ["ADJ-017", "ADJ-018"]
PA = [f"PA-{n:03}" for n in (1, 2, 3, 4, 9, 10, 11, 12, 19, 20, 21, 22)]
MODES = ("deductible_then_coinsurance", "copay", "no_cost", "coinsurance_only")
# Source readings, not alternative payout labels. See README for the criterion.
REFERENCE_MODE = dict(zip(BEN + ADJ, [
    MODES[0], MODES[0], MODES[0], MODES[1], MODES[2], MODES[0],
    MODES[1], MODES[0], MODES[0], MODES[3], MODES[1], MODES[1],
    MODES[0], MODES[2],
]))
BEN_FIELDS = ("member_responsibility", "plan_paid", "deductible_applied",
              "copay_amount", "coinsurance_amount", "ending_member_deductible",
              "ending_member_oopm")
EXTRACT = ("allowed", "deductible_credited", "oop_credited", "deductible_limit",
           "oop_limit", "copay", "coinsurance_percent")
DECISIONS = ("approve", "deny", "pend", "not_applicable")
COMMON = ("Apply only the supplied policy and case facts. Use the allowed amount. "
          "For the mode, identify the underlying service benefit before balance caps. "
          "If a governing document overrides a benefit, choose the mode after that override. "
          "Return only the requested fields, with no explanation.")


def load_case(case_id):
    url = f"{BASE}/items/{case_id}"
    cache = ROOT / ".build/pages" / f"{case_id}.html"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=40) as response:
            cache.write_bytes(response.read())
    page = cache.read_bytes()
    section = page.decode("utf-8").split("What the model was given", 1)[1].split("Every model", 1)[0]
    blocks = [unescape(re.sub(r"<[^>]+>", "", block))
              for block in re.findall(r"<code>(.*?)</code>", section, re.S)]
    assert len(blocks) == 3, (case_id, len(blocks))
    context, request, gold_text = blocks
    gold = json.loads(gold_text)
    assert gold["kind"] == ("decision" if case_id in PA else "fields")
    return dict(id=case_id, url=url, page_sha256=sha256(page).hexdigest(),
                input_sha256=sha256((context + "\n" + request).encode()).hexdigest(),
                context=context, request=request, gold=gold)


def schema_and_instruction(case_id, method):
    if case_id in PA:
        return ({
            "decision": {"type": "string", "enum": DECISIONS},
            "criteria": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        }, "Apply inclusion and exclusion criteria. Use pend when the policy requires "
            "missing documentation and no established reason defeats the request. "
            "Return the decision and the deciding policy clause IDs. For pend return missing documentation IDs.")
    properties = {"mode": {"type": "string", "enum": MODES}}
    if case_id in ADJ:
        properties.update(governing_document={"type": "string", "enum": ["certificate", "sbc", "mandate"]},
                          governing_section={"type": "string"})
    fields = EXTRACT if method == "hybrid" else BEN_FIELDS if case_id in BEN else BEN_FIELDS[:2]
    properties.update({field: {"type": "number"} for field in fields})
    if method == "hybrid":
        task = ("Extract source numbers only, without computing residual balances or payments. "
                "Amounts are dollars. Use the credited balances, thresholds, service copay, "
                "and member coinsurance percent on a 0 to 100 scale. Use 0 for irrelevant "
                "copay or rate. If an ADJ case omits an out-of-pocket limit, use 999999 "
                "and credited amount 0 to represent no specified cap. If coinsurance is "
                "unspecified and no dollars remain after deductible, use rate 0.")
    else:
        task = "Calculate the final member and plan amounts and any requested component amounts or ending balances."
    return properties, COMMON + " " + task


def infer(case, method):
    properties, instruction = schema_and_instruction(case["id"], method)
    schema = dict(type="object", properties=properties,
                  required=list(properties), additionalProperties=False)
    # Original task stays intact. The hybrid instruction explicitly changes the reply contract.
    body = dict(model="local-qwen", temperature=0, seed=42, max_tokens=320,
                response_format=dict(type="json_schema", json_schema=dict(name="parity_answer", strict=True, schema=schema)),
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": case["context"] + "\n\n" + case["request"]
                     + "\n\nFor this experiment reply ONLY with these fields: " + ", ".join(properties) + "."},
                ])
    post = Request("http://127.0.0.1:8312/v1/chat/completions",
                   data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(post, timeout=180) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    choice = result["choices"][0]
    raw = choice["message"]["content"]
    try:
        prediction = json.loads(raw)
        if not isinstance(prediction, dict) or set(prediction) != set(properties):
            prediction = None
    except (ValueError, TypeError):
        prediction = None
    return dict(id=case["id"], method=method, seconds=elapsed, raw=raw,
                finish=choice["finish_reason"], prediction=prediction)


def dollars(value):
    return Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def calculate(fields):
    allowed, deductible_ytd, oop_ytd, deductible_limit, oop_limit, copay, rate = (
        dollars(fields[key]) for key in EXTRACT
    )
    if min(allowed, deductible_ytd, oop_ytd, deductible_limit, oop_limit, copay, rate) < 0 or rate > 100:
        raise ValueError("Out of range")
    mode = fields["mode"]
    available = max(0, oop_limit - oop_ytd)
    deductible = min(allowed, available, max(0, deductible_limit - deductible_ytd)) if mode == MODES[0] else 0
    copay = min(allowed, available, copay) if mode == MODES[1] else 0
    coinsurance = min(available - deductible, dollars((allowed - deductible) * rate / 100)) if mode in (MODES[0], MODES[3]) else 0
    member = deductible + copay + coinsurance
    amounts = (member, allowed - member, deductible, copay, coinsurance, deductible_ytd + deductible, oop_ytd + member)
    return dict(zip(BEN_FIELDS, map(float, amounts)))


def run(output):
    with ThreadPoolExecutor(8) as threads:
        cases = list(threads.map(load_case, BEN + ADJ + PA))
    provenance = [{key: value for key, value in case.items() if key not in ("context", "request", "gold")}
                  for case in cases]
    output.mkdir(exist_ok=True)
    (output / "sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            for method in (("documentation",) if case["id"] in PA else ("direct", "hybrid")):
                row = infer(case, method)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(row["id"], row["method"], row["finish"], round(row["seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    run(parser.parse_args().output)
