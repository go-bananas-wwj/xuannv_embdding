# RSE V5 Execution Log

## Registered Scope

- Protocol: `v5_osm_assisted`
- Encoder family: `full_150` only
- Registered Xuannv encoders: **0 / 5 complete**
- Registered Xuannv Conv3x3 probes: **0 / 90 complete**
- Queue state: launched; folds 0--2 are running and folds 3--4 remain queued.
- Evidence rule: an in-progress attempt is not a paper result. A fold becomes admissible only
  after its selected checkpoint, export, and downstream artifacts pass the registered
  admission checks.

| Fold | Registered encoder | Current attempt | Checkpoint SHA-256 | Export registry entry | Xuannv probes (of 18) | Status |
| --- | --- | --- | --- | --- | ---: | --- |
| 0 | `paper_registered_v5_full_150_fold0_20260726` | running attempt 2 | pending | pending | 0 | training |
| 1 | `paper_registered_v5_full_150_fold1_20260726` | running attempt 2 | pending | pending | 0 | training |
| 2 | `paper_registered_v5_full_150_fold2_20260726` | attempt 1, recovery 2 | pending | pending | 0 | training |
| 3 | `paper_registered_v5_full_150_fold3_20260726` | pending | pending | pending | 0 | not launched |
| 4 | `paper_registered_v5_full_150_fold4_20260726` | pending | pending | pending | 0 | not launched |

## Attempt Rules

The queue runs only when its own script is Git-tracked, byte-identical to `HEAD`, and clean in
its owned worktree; it changes to the repository root before validation or training. Each encoder writes only under
`/data/xuannv_embedding/outputs/paper_registered_v5_20260726/<experiment>/attempt_N/`.
`attempt_manifest.json`, checkpoint-verification records, terminal status events, and final
logs are write-once. A fresh launch writes `logs/attempt_N.log`; every explicit recovery writes
the next immutable `attempt_N/logs/resume_K.log` and never replaces the fresh-attempt log.
`canonical_attempt.json` is created only after the selected attempt's verified checkpoint hash
matches; failed attempts remain preserved. An explicit recovery may resume only from a verified
checkpoint snapshot inside the same attempt.

## Deferred Work

AEF baseline work is explicitly deferred to **Task5**. No AEF embedding export or AEF probe is
included in this Task4 queue or in the Xuannv `0 / 90` status above.
