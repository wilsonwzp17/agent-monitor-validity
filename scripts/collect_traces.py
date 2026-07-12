"""Trace-collection stub for the candidate pool.

Implements the written inclusion rules for the candidate trace pool.
The full pool, with per-trace provenance, is committed to this repository
BEFORE preregistration. The study sample is then drawn from the pool by
the data & release lead by executing the preregistered selection rules;
the mentor does not draw the sample (see CHARTER.md, independence rules).

Planned sources
---------------
Natural stratum (target: at least 60-70% of the pool):
  - SWE-bench leaderboard trajectories
  - GAIA rollouts
  - WebArena / OSWorld episodes
Synthetic minority stratum (clearly marked, analyzed separately):
  - AgentTelemetry fault-injection scenarios

Inclusion rules (v0, finalized before pool collection)
------------------------------------------------------
1. Public source that is redistributable, or referencable by stable pointer.
2. Complete trace (no truncated rollouts) up to the length cap.
3. Length cap: [TO SET] tokens, with stratified sampling by length.
4. Provenance fields captured per trace: source, source_url,
   collection_date, length_tokens, stratum, task_family
   (see codebook/CODEBOOK_v0.md).
"""

import argparse

SOURCES = {
    "swebench": {"stratum": "natural", "url": "[TO SET: leaderboard trajectory source]"},
    "gaia": {"stratum": "natural", "url": "[TO SET]"},
    "webarena": {"stratum": "natural", "url": "[TO SET]"},
    "osworld": {"stratum": "natural", "url": "[TO SET]"},
    "agenttelemetry": {"stratum": "synthetic", "url": "[TO SET: fault-injection scenarios]"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the candidate trace pool under the written inclusion rules."
    )
    parser.add_argument("--out", default="pool/", help="Output directory for the pool")
    parser.add_argument("--source", choices=sorted(SOURCES), help="Collect a single source")
    parser.parse_args()
    raise NotImplementedError(
        "Pool collection lands before preregistration; see module docstring for the rules."
    )


if __name__ == "__main__":
    main()
