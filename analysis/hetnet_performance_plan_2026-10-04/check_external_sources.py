"""Fetch cited primary sources and print byte identities; never writes files.

Run with Python 3. PDF content is checked against the prior paper manifest.
Live documentation hashes describe a retrieval, not an immutable publication.
This does not execute a policy or validate experimental scientific conclusions.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("P1", "https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf", "paper"),
    ("P2", "https://raw.githubusercontent.com/CORE-Robotics-Lab/HetNet/bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec/AAMAS_22___HetNet_Supplementary.pdf", "pinned-paper"),
    ("W1", "https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/torch/profiler/profiler.py", "v2.2.1"),
    ("W2", "https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/docs/source/notes/randomness.rst", "v2.2.1"),
    ("W3", "https://slurm.schedmd.com/cpu_management.html", "live-documentation"),
    ("W4", "https://slurm.schedmd.com/sacct.html", "live-documentation"),
    ("W5", "https://www.dgl.ai/dgl_docs/en/2.1.x/generated/dgl.DGLGraph.local_scope.html", "2.1.x"),
    ("W6", "https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/torch/utils/benchmark/utils/timer.py", "v2.2.1"),
)
TOKENS = {
    "P1": (b"%PDF-",), "P2": (b"%PDF-",),
    "W1": (b"record_shapes", b"references to the tensors"),
    "W2": (b"not guaranteed", b"CPU and GPU"),
    "W3": (b"cpu-bind", b"nomultithread"),
    "W4": (b"MaxRSS", b"TotalCPU"),
    "W5": (b"local_scope", b"Inplace operations"),
    "W6": (b"warmups", b"num_threads"),
}


def main():
    papers = json.loads((ROOT / "research/papers/TRAINING_PAPER_MANIFEST.json").read_text())["papers"]
    expected = {"P1": papers[0]["sha256"], "P2": papers[1]["sha256"]}

    def fetch(source):
        reference, url, version = source
        request = Request(url, headers={"User-Agent": "HetNet-evidence-check/1.0"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
            result = {"reference": reference, "url": url, "resolved_url": response.geturl(),
                      "version": version, "http_status": response.status,
                      "content_type": response.headers.get("Content-Type"),
                      "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        result["expected_content_markers_present"] = all(token in payload for token in TOKENS[reference])
        if reference in expected:
            result["prior_manifest_sha256"] = expected[reference]
            result["matches_prior_manifest"] = result["sha256"] == expected[reference]
        result["passed"] = result["expected_content_markers_present"] and result.get("matches_prior_manifest", True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, SOURCES))
    result = {"schema_version": 1, "retrieved_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
              "command": "python3 analysis/hetnet_performance_plan_2026-10-04/check_external_sources.py",
              "status": "passed" if all(row["passed"] for row in results) else "failed",
              "scope": "Primary-source byte retrieval and markers; paper hashes match the prior manifest. "
                       "This checker does not inspect local PDF copies. Documentation content must still be interpreted at the cited sections; "
                       "markers and hashes alone do not prove any scientific claim. Live Slurm documentation can change.",
              "sources": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
