"""Record output hashes for a baseline built before the pipeline recorded them.

The pipeline now hashes ``wf_scored.parquet`` and ``execution_prices.parquet``
as it writes them. Baselines built earlier have no such record, so the verifier
refuses them -- correctly: an unhashed output cannot be vouched for, and
quietly passing it is how a forged score file came to print a 98% CAGR under
"all 12 gates passed".

Rebuilding is the honest fix and takes about forty minutes. Sealing is the
cheap one, and it buys strictly less:

    recorded-at-write    these bytes are what the run produced
    sealed-after-the-fact these bytes are what was on disk when someone sealed

A seal is tamper-evident from the seal onward and proves nothing before it. The
verifier prints the distinction rather than treating the two alike, and this
script will not overwrite a hash that was recorded at write time.

One thing is checked rather than assumed: the execution panel must still be the
pivot of the hashed snapshot it came from. That is real provenance and it is
available retroactively, so sealing it is not a blind act. No such check exists
for the scores -- nothing in a baseline can re-derive a model's output -- which
is exactly why the hash needed to be recorded at write time.

    uv run python scripts/seal_baseline_outputs.py artifacts/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_baseline import OUTPUTS, gate_execution_derivation  # noqa: E402

from stock_predictor import repro  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Re-seal outputs that already carry a hash. Refuses "
                         "recorded-at-write hashes regardless.")
    args = ap.parse_args()

    d = args.baseline_dir
    man_path = d / "snapshot" / "manifest.json"
    if not man_path.exists():
        sys.exit(f"no manifest at {man_path}")
    man = json.loads(man_path.read_text())

    derivation = gate_execution_derivation(d)
    print(derivation.report())
    if not derivation.passed:
        sys.exit("\nThe execution panel does not derive from its snapshot. "
                 "Sealing it would certify bytes that are already known to be "
                 "wrong. Rebuild instead.")

    outputs = man.setdefault("outputs", {})
    sealed = []
    for name in OUTPUTS:
        path = d / f"{name}.parquet"
        if not path.exists():
            sys.exit(f"missing {path}")
        existing = outputs.get(name)
        if existing:
            prov = str(existing.get("provenance", "unknown"))
            if prov == "recorded-at-write":
                print(f"{name}: already recorded at write time, leaving alone")
                continue
            if not args.force:
                print(f"{name}: already sealed ({prov}); pass --force to re-seal")
                continue
        meta = repro.register_output(man, name, path)
        meta["provenance"] = "sealed-after-the-fact"
        meta["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta["sealed_at_commit"] = repro.git_revision().get("commit")
        sealed.append(name)
        print(f"{name}: sealed {meta['sha256'][:16]}")

    if not sealed:
        print("\nNothing to seal.")
        return

    repro.write_manifest(man_path, man)
    print(f"\nSealed {len(sealed)} output(s) into {man_path}")
    print("These artifacts are now tamper-evident. They are not reproduced; "
          "only a rebuild does that.")


if __name__ == "__main__":
    main()
