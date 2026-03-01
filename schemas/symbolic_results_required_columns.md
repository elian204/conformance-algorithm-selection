# Symbolic Results CSV Contract

This file defines the required schema for symbolic baseline outputs so they can be joined with A* experiment results.

## Required Columns

- `model_id` (string)
  - Stable model identifier matching the `model_id` from A* `results.csv`.
- `trace_hash` (string)
  - Trace content hash matching A* `trace_hash`.
- `symbolic_time_seconds` (float, >= 0)
  - Wall-clock or CPU time for symbolic alignment runtime.
- `symbolic_status` (string)
  - Allowed values: `ok`, `timeout`, `error`, `no_solution`, `max_expansions`.

## Optional Columns

- `trace_id` (string)
  - Optional extra disambiguation key.
- `symbolic_expansions` (int)
- `symbolic_generations` (int)
- `symbolic_memory_mb` (float)
- Any additional symbolic diagnostics.

## Key Semantics

- Primary join key is:
  - `model_id + trace_hash`
- If both tables include `trace_id`, join can use:
  - `model_id + trace_hash + trace_id`

## Validation

Use:

```bash
python scripts/validate_symbolic_results.py --input path/to/symbolic_results.csv
```

For strict `trace_id` requirement:

```bash
python scripts/validate_symbolic_results.py --input path/to/symbolic_results.csv --require-trace-id
```
