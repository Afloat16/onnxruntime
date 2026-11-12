import ast
import hashlib
import json
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit

original_source = subprocess.check_output(
    ["git", "show", "a8246c0d094f1ab0d6bf236228e5e67683321cd7:onnxruntime/python/tools/quantization/quant_utils.py"],
    text=True,
)
node = next(x for x in ast.parse(original_source).body if isinstance(x, ast.FunctionDef) and x.name == "pack_bytes_to_4bit")
namespace = {"numpy": np}
exec(compile(ast.Module(body=[node], type_ignores=[]), "<exact upstream pack_bytes_to_4bit>", "exec"), namespace)
original = namespace["pack_bytes_to_4bit"]
rows = []
for length in (32, 4096, 1000000, 8000000):
    source = bytes(range(16)) * (length // 16)
    assert len(source) == length
    expected = original(source)
    assert pack_bytes_to_4bit(source) == expected
    row = {"elements": length, "packed_sha256": hashlib.sha256(expected).hexdigest()}
    for label, function in (("original", original), ("vectorized", pack_bytes_to_4bit)):
        function(source)
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            output = function(source)
            timings.append((time.perf_counter() - start) * 1000)
            assert output == expected
        del output
        tracemalloc.start()
        output = function(source)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert output == expected
        row[label] = {"median_ms": statistics.median(timings), "samples_ms": timings, "peak_traced_bytes": peak}
    rows.append(row)
result = {"warmups": 1, "repeats": 5, "scope": "Packing helper only; input created outside measurements; tracemalloc separate from timings.", "rows": rows}
Path("ort-packing-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
