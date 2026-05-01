# Known Issues

Issues discovered during benchmarking that are not yet fixed in the library.

---

## 1. `_upsert_vec` silently drops vectors on sqlite-vec INSERT failure

**File:** `memweave/store.py:959-960`  
**Status:** Open  
**Severity:** Medium — silent degraded retrieval, no user-visible error  

### What happens

`_upsert_vec` inserts chunk embeddings into the sqlite-vec virtual table
(`chunks_vec`). The INSERT is wrapped in a bare `except Exception` that only
logs at `DEBUG` level:

```python
except Exception as exc:
    logger.debug("Failed to upsert vec for %s: %s", chunk_id, exc)
```

If the INSERT fails for any reason, the chunk gets no vector and becomes
invisible to all vector and hybrid searches. There is no warning, no error
propagated to the caller, and no indication to the user that retrieval is
degraded.

### Evidence

During a LongMemEval benchmark pre-run (question `b320f3f8`, 46 sessions,
84 total chunks), `embeddings_computed=63` — meaning 21 chunks had no entry
in `chunks_vec`. No `WARNING` output appeared during the full 50-question
dev run, confirming the embedding API itself did not fail (that path logs at
`WARNING`). The silent failure is therefore in `_upsert_vec`.

### Impact on benchmark

The dev split (50 questions) returned **Recall@5 = 94.0%**, within the
expected 93–98% range, so the issue is not catastrophic. It may be silently
costing 1–3 questions in the held-out 450-question run.

Any library user who hits this bug gets silently degraded retrieval — sessions
that were indexed appear to be missing — with no indication of what went wrong.

### Likely causes

- Vector dimension mismatch between the table schema and a returned embedding
- sqlite-vec internal error on a specific chunk's binary payload
- Concurrency issue if the virtual table is accessed before `ensure_vector_table`
  fully completes

### Fix

Promote the log level from `DEBUG` to `WARNING` at minimum so failures
surface. Ideally re-raise after logging so callers can decide how to handle:

```python
# store.py:959
except Exception as exc:
    logger.warning("Failed to upsert vec for %s: %s", chunk_id, exc)
    # optionally: raise
```

A more thorough fix would add a post-index integrity check: compare
`SELECT COUNT(*) FROM chunks` against `SELECT COUNT(*) FROM chunks_vec`
and warn if they diverge.
