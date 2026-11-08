# Public API and stability

ipax is in beta (`0.x`). This page says which names the project treats as its
public API, how changes to them are announced, and what `1.0.0` will
promise.

## What is public

- Every name in `ipax.__all__`, imported from the top-level package:
  `ipax.solve`, `ipax.Problem`, `ipax.Options` and its option groups,
  `ipax.Result` and the other result types, the sparse operators
  `ipax.COOOperator`/`CSROperator`/`CSCOperator`, and the exceptions and
  warnings.
- The front ends in `ipax.interfaces`, at the paths their modules list in
  `__all__` (today `ipax.interfaces.scipy.minimize_ipax` and `STATUS_CODES`).
- The extension points documented in the [API reference](reference.md), at
  the import path shown there: `LinearOperator` and its documented methods,
  `as_operator`, `MatrixFreeJacobian` and `ScaledOperator` in
  `ipax.backend.operators`; the `LinearSolver` protocol, `LinearSolveError`
  and `numerical_failure_types` in `ipax.linalg.solver`; `array_namespace`,
  `capabilities` and `Capabilities` in `ipax.backend.namespace`.

## What is not

Everything else may change in any release without notice:

- other modules and their contents (`ipax.ipm.*`, `ipax.linalg.*`,
  `ipax.problem.*`, `ipax.backend.*` beyond the entries above), including the
  `__all__` lists inside them;
- names starting with an underscore, private attributes of classes you
  subclass, and undocumented methods or hooks;
- the sparse-adapter protocol in `ipax.backend.sparse`;
- the wording of free-text fields: `Result.message`, `Result.linear_solver`
  and the strings in `Result.routes`;
- `ipax.testing`, which holds test and benchmark oracles and is provisional.

## How changes are announced

Every user-visible change is listed in the
[changelog](https://github.com/wahln/ipax/blob/main/CHANGELOG.md): additions
under *Added*, behaviour changes under *Changed*, renames under *Deprecated*,
and removals under *Removed*.

A public name that is renamed or removed keeps working for at least one minor
release and warns when used:

- `FutureWarning` for names users call or construct (functions, classes,
  option fields, option values). Python shows it by default.
- `DeprecationWarning` for hooks only extension authors implement (methods of
  `LinearOperator` or `LinearSolver` that the solver calls). Python hides it
  outside `__main__` and test runners.

Options marked **Experimental** in their documentation are levers whose
benefit is still being measured on the benchmark corpus. Their semantics or
defaults may change in a minor release; they are still removed only after a
deprecation period.

Until `1.0.0`, a behaviour change that cannot carry a deprecation period
(for example a field whose meaning changes) can ship in a minor release. It is
listed under *Changed* with what code needs to do.

## Errors and warnings

Everything ipax raises on purpose derives from `ipax.IpaxError` and from the
built-in exception it raised before 0.14, so existing `except` clauses keep
working:

| Exception | Also a | Raised when |
|---|---|---|
| `ConfigurationError` | `ValueError` | an option has an invalid value |
| `ProblemDefinitionError` | `ValueError` | `x0` or the bounds do not match `n_vars`, or `Problem.provides()` names an unknown method |
| `DerivativeUnavailableError` | `RuntimeError` | a derivative the solve needs cannot be obtained |
| `LinearSolveError` | `RuntimeError` | a linear solve failed in a way the solver could not recover from |

An argument of the wrong type, such as an `options` dict, raises a plain
`TypeError`. Numerical trouble during the iterations is not raised: the solve
returns a `Result` whose `status` says how it ended. An exception raised by
your own callbacks propagates out of `ipax.solve` unchanged.

Warnings derive from `ipax.IpaxWarning`. `ipax.FiniteDifferenceWarning`
reports a derivative approximated by finite differences; silence it with
`warnings.simplefilter("ignore", ipax.FiniteDifferenceWarning)`.
