# SWE25 editable Pi optimization results

Solver: `openrouter/openai/gpt-5.6-luna`. Proposer: Codex `gpt-6-astra`.
Ten rounds, one candidate per round, four concurrent tasks, fixed 25-task SWE training batch.

Best: round 5, 18/25 (72%); baseline: 14/25 (56%). Rounds 9 and 10 tied the best; ties retain the earlier winner.

These are training rewards, not held-out performance or evidence about reward hacking. No held-out evaluation was performed.

| Candidate | Passed | Errors | Valid |
| --- | ---: | ---: | --- |
| baseline | 14/25 | 0 | True |
| iter_001_candidate_00 | 0/25 | 25 | False |
| iter_002_candidate_00 | 17/25 | 0 | True |
| iter_003_candidate_00 | 16/25 | 0 | True |
| iter_004_candidate_00 | 16/25 | 0 | True |
| iter_005_candidate_00 | 18/25 | 0 | True |
| iter_006_candidate_00 | 17/25 | 0 | True |
| iter_007_candidate_00 | 16/25 | 0 | True |
| iter_008_candidate_00 | 16/25 | 0 | True |
| iter_009_candidate_00 | 18/25 | 0 | True |
| iter_010_candidate_00 | 18/25 | 0 | True |

## Contents

- `best.mjs`: final winning harness, identical to round 5.
- `optimization.json`: frozen models, task IDs, settings, and implementation hashes.
- `history/`: baseline and all ten candidate harnesses, proposer reflections, rewards, selections, and API context.
- `state.json` and `summary.json`: final state and compact scores.

Raw Inspect `.eval` logs, per-sample traces, proposer CLI transcripts, and the process lock are omitted (the original run is approximately 1.3 GB). Original `log_files` paths in reward files refer to the local raw run and are retained as provenance; those files are not included here.

The run was interrupted during round 5 and resumed; the archived round 5 reward comes from its completed reevaluation. This is an analysis snapshot, not a complete resume directory.
