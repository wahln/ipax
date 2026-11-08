# API reference

Generated from docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/).
Everything documented here is public (see
[Public API and stability](stability.md)). Names exported from the top-level
package are imported as `ipax.<name>` — e.g. `ipax.Result`, whose entry below
shows its defining module `ipax.result`. The SciPy front end, the
linear-algebra extension points and the backend introspection are imported
from the module shown in their entry.

## Entry point

The single function most users call. It resolves derivatives, selects a linear
solver, runs the interior-point loop, and returns a [`Result`](#results-and-diagnostics).

::: ipax.solve.solve

## SciPy front end

`scipy.optimize.minimize`-compatible entry point (needs `ipax[scipy]`); see
[Coming from SciPy](guide/scipy.md).

::: ipax.interfaces.scipy.minimize_ipax

::: ipax.interfaces.scipy.STATUS_CODES

## Problem definition

Define a model by subclassing [`Problem`](#ipax.problem.base.Problem), or use one
of the ready-made adapters for callable-based, quadratic, and linear programs.

::: ipax.problem.base.Problem

::: ipax.problem.function.FunctionProblem

::: ipax.problem.function.QuadraticProblem

::: ipax.problem.function.LinearProblem

## Options

`Options` is the top-level configuration object; the remaining dataclasses
configure individual subsystems and are attached to it. All are frozen and
validated on construction.

::: ipax.options.Options

::: ipax.options.OptimalityConditionOptions

::: ipax.options.AcceptableStoppingOptions

::: ipax.options.ScalingOptions

::: ipax.options.CorrectionsOptions

::: ipax.options.BarrierOptions

::: ipax.options.LineSearchOptions

::: ipax.options.RegularizationOptions

::: ipax.options.LBFGSOptions

::: ipax.options.KrylovOptions

::: ipax.options.RestorationOptions

::: ipax.options.SparseOptions

::: ipax.options.BreedveldOptions

::: ipax.options.DenseOptions

## Results and diagnostics

What a solve returns, plus the per-iteration snapshot passed to a callback.

::: ipax.result.Result

::: ipax.result.Status

::: ipax.result.KKTResiduals

::: ipax.result.DerivativeSources

::: ipax.result.Routes

::: ipax.result.IterationRecord

::: ipax.result.IterationInfo

::: ipax.result.IterationCallback

## Warm starting

::: ipax.result.WarmStart

## Errors and warnings

The documented error conditions raise these classes; each derives from
`IpaxError` and from the built-in ipax raised before 0.14 (see
[Public API and stability](stability.md#errors-and-warnings)).

::: ipax.errors.IpaxError

::: ipax.errors.ConfigurationError

::: ipax.errors.ProblemDefinitionError

::: ipax.errors.DerivativeUnavailableError

::: ipax.errors.UnsupportedBackendError

::: ipax.errors.LinearSolveError

::: ipax.errors.IpaxWarning

::: ipax.errors.FiniteDifferenceWarning

## Sparse operators

Jacobians or constraint matrices with explicit sparsity, from COO triplets or
compressed rows/columns. A `scipy.sparse` matrix `A` becomes
`ipax.CSROperator(A.indptr, A.indices, A.data, A.shape)` after `A = A.tocsr()`
(note the argument order differs from SciPy's `csr_matrix((data, indices,
indptr))`).

::: ipax.backend.operators.COOOperator

::: ipax.backend.operators.CSROperator

::: ipax.backend.operators.CSCOperator

## Linear-algebra extension points

Advanced surface for supplying structured or matrix-free operators. The core
only ever sees these protocol types — see [invariants #1–#4](architecture.md).

::: ipax.backend.operators.LinearOperator
    options:
      members: [shape, matvec, rmatvec, matmat, diagonal, to_coo, gram_diagonal]

::: ipax.backend.operators.as_operator

::: ipax.backend.operators.MatrixFreeJacobian

::: ipax.backend.operators.ScaledOperator

::: ipax.linalg.solver.LinearSolver

::: ipax.linalg.solver.numerical_failure_types

## Backend introspection

::: ipax.backend.namespace.array_namespace

::: ipax.backend.namespace.capabilities

::: ipax.backend.namespace.Capabilities
