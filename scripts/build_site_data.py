"""Collect saved benchmark numbers into docs/data/results.json (read by the web demo and the report)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
eff = json.loads((R / "efficiency.json").read_text())
labels = {"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer V6"}
paper_f1 = {"tinycd": 91.05, "bit": 89.31, "changeformer": 90.40}
models = []
for k, name in labels.items():
    ev = json.loads((R / f"eval_{k}_test.json").read_text())
    e = eff["models"][k]
    models.append(dict(id=k, model=name, **ev["metrics"], paper_f1=paper_f1[k], params=e["params_used"],
                       gflops=e["gflops"], cpu_ms=e["latency_cpu_ms"]["median"]))
samples = [dict(dir=s["dir"], title=s["title"], f1={k: v["f1"] for k, v in s["models"].items()})
           for s in json.loads((ROOT / "docs/samples/index.json").read_text())]
out = ROOT / "docs" / "data" / "results.json"
out.write_text(json.dumps(dict(models=models, samples=samples,
                               protocol="LEVIR-CD256 test split, 2,048 pairs, threshold 0.5, released checkpoints"), indent=1))
print(out)
