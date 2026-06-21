# API reference

Generated from docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/).
Everything documented here is importable directly from the top-level `ipax`
package; the fully-qualified module path is shown for reference only.

## Entry point

The single function most users call. It resolves derivatives, selects a linear
solver, runs the interior-point loop, and returns a [`Result`](#results-and-diagnostics).

::: ipax.solve.solve

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

::: ipax.options.BreedveldOptions

## Results and diagnostics

What a solve returns, plus the per-iteration snapshot passed to a callback.

::: ipax.result.Result

::: ipax.result.Status

::: ipax.result.KKTResiduals

::: ipax.result.DerivativeSources

::: ipax.result.IterationRecord

::: ipax.result.IterationInfo

::: ipax.result.IterationCallback

## Warm starting

::: ipax.result.WarmStart

## Linear-algebra extension points

Advanced surface for supplying structured/matrix-free operators or custom solve
strategies. The core only ever sees these protocol types — see
[invariants #1–#4](architecture.md).

::: ipax.backend.operators.LinearOperator
    options:
      members: [shape, matvec, rmatvec, matmat, diagonal, to_coo, gram_diagonal]

::: ipax.backend.operators.as_operator

::: ipax.linalg.solver.LinearSolver

::: ipax.linalg.solver.LinearSolveError

::: ipax.linalg.solver.select_solver

## Backend introspection

::: ipax.backend.namespace.array_namespace

::: ipax.backend.namespace.capabilities

::: ipax.backend.namespace.Capabilities
