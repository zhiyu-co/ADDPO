#!/usr/bin/env python3
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.input, encoding="utf-8") as reader:
    rows = [json.loads(line) for line in reader if line.strip()]

for row in rows:
    row["model_prediction"] = row.get("answer_prediction", "")

with open(args.output, "w", encoding="utf-8") as writer:
    json.dump(rows, writer, ensure_ascii=False)
