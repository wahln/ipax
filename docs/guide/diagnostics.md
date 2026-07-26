# Monitoring & re-solving

Three facilities let you watch a solve, react to it, and reuse its work:
layered **logging**, per-iteration **callbacks**, and **warm starting**.

## Logging

`ipax` logs through a standard-library logger named `"ipax"` carrying a
`NullHandler`, so importing the package never prints anything on its own. There
are two ways to see output.

### Quick: `verbose`

Set `Options.verbose` (an integer `0`–`6`) to attach a console handler with
progressively more detail. Each level adds a content tier on top of the ones
below it:

| `verbose` | Adds |
|:---:|---|
| `0` | silent (only warnings/errors) |
| `1` | result summary |
| `2` | + per-iteration table & timing split |
| `3` | + problem structure |
| `4` | + resolved solver setup |
| `5` | + every sub-option |
| `≥6` | + debug diagnostics |

```python
res = ipax.solve(problem, x0, options=ipax.Options(verbose=2))
```

The per-iteration table (level ≥ 2) shows the objective, infeasibility, KKT
error, `mu`, step lengths, applied regularization, and the time split between
your problem callbacks and the inner solves — the first place to look when a
solve is slow.

### Full control: own the logger

Because every record is emitted regardless of `verbose`, an application can
attach its own handlers/formatters/filters to the `"ipax"` logger and keep full
control, instead of using the built-in console handler:

```python
import logging

logging.getLogger("ipax").setLevel(logging.DEBUG)
logging.getLogger("ipax").addHandler(my_handler)
# leave Options.verbose at 0 so ipax does not add its own handler
```

## Iteration callbacks

Pass `callback=` to `solve` to receive a read-only
[`IterationInfo`](../reference.md#ipax.result.IterationInfo) snapshot once per
iteration. It carries the iteration `record` (the same one appended to
`res.history`) plus the current primal/dual iterate (`x`, `s`, `y_eq`,
`y_ineq`, `z_lower`, `z_upper`; `None` for absent blocks), all in the original
problem's units.

```python
def monitor(info):
    print(info.record.iteration, info.record.kkt_error)


res = ipax.solve(problem, x0, callback=monitor)
```

### Custom stopping

Returning a **truthy** value from the callback stops the solve early with
[`Status.STOPPED`](results.md#status). This implements arbitrary stopping rules —
a wall-clock budget, a target objective, an external cancellation flag:

```python
def stop_when_good_enough(info):
    return info.record.objective < 1e-3  # stop as soon as f drops below 1e-3


res = ipax.solve(problem, x0, callback=stop_when_good_enough)
assert res.status is ipax.Status.STOPPED
```

Treat the arrays as read-only and copy before mutating. For standard
convergence/budget limits prefer [the termination options](options.md#termination);
reserve callbacks for rules the built-in conditions can't express.

## Warm starting

When you re-solve a *perturbed* problem — a re-plan, a parameter sweep, a
continuation — seeding the next solve with the previous solution's multipliers
saves iterations. The primal point comes from the `x0` you pass; the dual
quantities come from a [`WarmStart`](../reference.md#ipax.result.WarmStart).

The common case reuses a prior `Result` directly:

```python
res1 = ipax.solve(problem, x0)

# ... perturb the problem slightly ...

res2 = ipax.solve(
    problem_perturbed,
    x0=res1.x,  # start from the old solution
    warm_start=ipax.WarmStart.from_result(res1),  # reuse its multipliers
)
```

Any field left `None` falls back to the standard μ-complementarity
initialization, so a partial warm start is fine. Slacks aren't stored on
`Result` (they're recomputed from feasibility); supply them explicitly via
`WarmStart.from_result(res1, s=...)` or the `WarmStart(s=...)` constructor if you
have them. Slacks and bound/inequality multipliers are floored to stay strictly
interior; equality multipliers (free sign) pass through unchanged.

Warm-start values are in the original problem's units; with
[scaling](options.md#problem-scaling) enabled the solver rescales them
internally to match the scaled subproblem. The payoff is real — re-solving from
a converged point with its duals typically reaches the tolerance in noticeably
fewer iterations than a cold start, and the duals (not just `x0`) do most of
that work.
