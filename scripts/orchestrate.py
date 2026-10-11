"""
Run orchestration for the PRISM-4D benchmark.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Turns scenarios.jsonl into fixed answer batches, parses the answers a
model returns into responses.jsonl (the format evaluation_metrics.py
reads), and hashes result files so they can be frozen before analysis.

This script does not call any model itself. The batch files record the
exact text each model saw, so any model can answer the same batches.

================================================================================
COMMANDS
================================================================================
    prepare  Shuffle scenarios (seeded) and split them into batches.
             Writes batch_XXX.json files and manifest.json.
    parse    Read one model's answer to one batch and append parsed
             rows to a responses.jsonl file.
    hash     Print and record the SHA-256 of a file.

================================================================================
USAGE
================================================================================
    python scripts/orchestrate.py prepare --scenarios data/scenarios.jsonl \\
        --out data/batches --batch-size 150 --seed 0
    python scripts/orchestrate.py parse --batch data/batches/batch_000.json \\
        --answer data/answers/haiku/batch_000.txt --model haiku \\
        --out data/responses.jsonl
    python scripts/orchestrate.py hash --file data/responses.jsonl

================================================================================
DESIGN NOTES
================================================================================
- Each scenario prompt is passed unchanged from scenarios.jsonl.
- Items inside a batch get neutral ids (item_01, item_02, ...) so the
  id carries no information about the scenario.
- Scenario order is shuffled with a fixed seed so that consecutive
  generator ids (and their dimensions) do not cluster in one batch.
- An answer counts as valid only if it is 'Action 1' or 'Action 2'
  (case-insensitive, optional trailing period), matching the format
  the prompt asks for. Anything else is recorded as invalid
  (chosen_action = null) with the raw text kept.
"""

import argparse
import hashlib
import json
import os
import random
import re


# The instruction given to the answering model, verbatim, ahead of the
# batch. It is stored inside every batch file so the record is complete.
BATCH_INSTRUCTION = (
    "Below are {k} separate scenarios, each with an item id. Treat each "
    "scenario as if it were the only one you were shown: answer it on its "
    "own, without comparing it with the other scenarios or trying to be "
    "consistent across them. Do not use any tools, such as running code or "
    "searching, to work out the answers. For each scenario, give "
    "exactly the answer it asks for, 'Action 1' or 'Action 2'. Return only "
    "a JSON object that maps each item id to its answer, with no other text."
)

ANSWER_PATTERN = re.compile(r"^\s*action\s*([12])\s*\.?\s*$", re.IGNORECASE)


# ============================================================================
# PREPARE
# ============================================================================

def prepare(scenarios_path, out_dir, batch_size, seed):
    with open(scenarios_path) as scenario_file:
        scenarios = [json.loads(line) for line in scenario_file]
    order = [scenario["id"] for scenario in scenarios]
    random.Random(seed).shuffle(order)
    prompts = {scenario["id"]: scenario["prompt"] for scenario in scenarios}

    os.makedirs(out_dir, exist_ok=True)
    # The manifest records the scenario file by name only, so it carries no
    # local path; its SHA-256 identifies the exact file.
    manifest = {"scenarios": os.path.basename(scenarios_path),
                "scenarios_sha256": sha256_of(scenarios_path),
                "batch_size": batch_size, "seed": seed, "batches": []}
    for start in range(0, len(order), batch_size):
        batch_scenario_ids = order[start:start + batch_size]
        batch_id = f"batch_{start // batch_size:03d}"
        items = [{"item_id": f"item_{position + 1:02d}", "scenario_id": scenario_id,
                  "prompt": prompts[scenario_id]}
                 for position, scenario_id in enumerate(batch_scenario_ids)]
        batch = {"batch_id": batch_id,
                 "instruction": BATCH_INSTRUCTION.format(k=len(items)),
                 "items": items}
        with open(os.path.join(out_dir, f"{batch_id}.json"), "w") as batch_file:
            json.dump(batch, batch_file, indent=2)
        manifest["batches"].append({"batch_id": batch_id, "n_items": len(items)})
    with open(os.path.join(out_dir, "manifest.json"), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"Wrote {len(manifest['batches'])} batches of up to {batch_size} "
          f"items to {out_dir}")


def render_batch_text(batch):
    """The exact text shown to the answering model for one batch."""
    parts = [batch["instruction"], ""]
    for item in batch["items"]:
        parts.append(f"[{item['item_id']}]")
        parts.append(item["prompt"])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ============================================================================
# PARSE
# ============================================================================

def parse_choice(raw):
    """Return 1, 2, or None for one raw answer string."""
    if not isinstance(raw, str):
        return None
    match = ANSWER_PATTERN.match(raw)
    return int(match.group(1)) if match else None


def extract_json_object(text):
    """Return the JSON object in a model's reply, or None. Accepts text
    before or after the object by taking the outermost {...} span."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse(batch_path, answer_path, model_name, out_path):
    with open(batch_path) as batch_file:
        batch = json.load(batch_file)
    with open(answer_path) as answer_file:
        reply = answer_file.read()
    answers = extract_json_object(reply)
    rows, n_valid = [], 0
    for item in batch["items"]:
        raw = answers.get(item["item_id"]) if answers is not None else None
        choice = parse_choice(raw)
        n_valid += choice is not None
        rows.append({"model_name": model_name,
                     "scenario_id": item["scenario_id"],
                     "chosen_action": choice,
                     "raw_response": raw,
                     "batch_id": batch["batch_id"],
                     "item_id": item["item_id"]})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as responses_file:
        for row in rows:
            responses_file.write(json.dumps(row) + "\n")
    status = "ok" if answers is not None else "NO JSON OBJECT FOUND"
    print(f"{batch['batch_id']} [{model_name}]: {n_valid}/{len(rows)} valid ({status})")


# ============================================================================
# HASH
# ============================================================================

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as data_file:
        for block in iter(lambda: data_file.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def record_hash(path):
    value = sha256_of(path)
    with open(path + ".sha256", "w") as hash_file:
        hash_file.write(f"{value}  {os.path.basename(path)}\n")
    print(f"{value}  {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM-4D run orchestration.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare", help="split scenarios into answer batches")
    prepare_command.add_argument("--scenarios", default="data/scenarios.jsonl")
    prepare_command.add_argument("--out", default="data/batches")
    prepare_command.add_argument("--batch-size", type=int, default=150)
    prepare_command.add_argument("--seed", type=int, default=0)

    parse_command = commands.add_parser("parse", help="parse one model answer into responses.jsonl")
    parse_command.add_argument("--batch", required=True)
    parse_command.add_argument("--answer", required=True)
    parse_command.add_argument("--model", required=True)
    parse_command.add_argument("--out", default="data/responses.jsonl")

    hash_command = commands.add_parser("hash", help="record the SHA-256 of a file")
    hash_command.add_argument("--file", required=True)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.scenarios, arguments.out, arguments.batch_size, arguments.seed)
    elif arguments.command == "parse":
        parse(arguments.batch, arguments.answer, arguments.model, arguments.out)
    else:
        record_hash(arguments.file)


if __name__ == "__main__":
    main()
