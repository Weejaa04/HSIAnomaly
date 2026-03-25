import json
import csv
import statistics

with open("benchmark_all.json", "r") as f:
    data = json.load(f)

foods = ["Almond", "Pistachio", "GarlicStems"]
metrics = ["n_params", "infer_time_sec", "max_vram_gb", "roc_auc", "pr_auc"]
headers = ["arch name", "param count", "infer time", "max vram", "auroc", "aupr"]

arch_names = list(data.keys())

for food in foods:
    rows = []
    for arch in arch_names:
        row = [arch]
        for m in metrics:
            row.append(data[arch][food][m])
        rows.append(row)

    with open(f"benchmark_{food}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

with open("benchmark_average.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for arch in arch_names:
        row = [arch]
        for m in metrics:
            values = [data[arch][food][m] for food in foods]
            avg = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            row.append(f"{avg} +- {stdev}")
        writer.writerow(row)

print(
    "Created 4 CSV files: benchmark_Almond.csv, benchmark_Pistachio.csv, benchmark_GarlicStems.csv, benchmark_average.csv"
)
