# The algorithm

A primal–dual log-barrier interior-point method.

## Standard form

Inequalities get non-negative slacks; bounds stay explicit:

$$
\min_x f(x)\quad\text{s.t.}\quad c(x)=0,\; g(x)+s=0,\; s\ge 0,\; x_L\le x\le x_U.
$$

The barrier subproblem for $\mu>0$ adds $-\mu\sum\ln(s_i)-\mu\sum\ln(x-x_L)-\mu\sum\ln(x_U-x)$.

## Newton step → condensed system

After eliminating the slack and bound-dual increments, the symmetric augmented
saddle system reduces to the **condensed normal-equations** matrix
(Breedveld eq. 18, generalized):

$$
N = W + \Sigma_x + \nabla g^\top (S^{-1}\Lambda)\,\nabla g,
$$

with $\Sigma_x=(X-X_L)^{-1}Z_L+(X_U-X)^{-1}Z_U$. With equalities present, the
$(\Delta x,\Delta y)$ saddle gets Friedlander–Orban primal–dual regularization
($\delta_w$ on the (1,1) block, $-\delta_c$ on the (2,2) block) so it is
quasidefinite and factorizable **without an inertia oracle**.

## Globalization

IPOPT-style **filter line search** on $(\theta,\varphi_\mu)$ with second-order
correction — solved on the step's own retained factorization at
unregularized steps (Wächter & Biegler §2.4, eq. (26)); at a
$\delta_w > 0$ step ipax deviates deliberately and re-solves fresh at
$\delta_w = 0$, which measurably corrects feasibility better — and a
**feasibility restoration** phase (Wächter & Biegler §2–3). A lighter
**Breedveld step controller** (Markov filter + ratio control) is selectable
for convex/RT-like problems.

## Higher-order corrections

Optionally, each outer iteration reuses its KKT factorization for extra
complementarity-target solves that better track the central path:

- **Mehrotra (1992)** predictor–corrector — an affine (predictor) solve sets an
  adaptive centering parameter and a second-order complementarity correction;
  one extra solve.
- **Gondzio (1996)** multiple centrality corrections — up to $K$ further solves
  pulling the complementarity products into a box $[\gamma\mu,\,\mu/\gamma]$,
  kept only while the step lengths keep growing.

Both reuse the *same* factorization (no refactor, consistent $\delta_w$) and
degenerate to the plain step when there are no inequalities or bounds. Enable
with `Options(corrections="mehrotra" | "gondzio")`.

## Defaults

- Fraction-to-boundary $\tau=\max(\tau_{\min},1-\mu)$.
- Monotone Fiacco–McCormick $\mu$ update.
- Optimality termination on the scaled KKT residual components — by default each
  of dual infeasibility, constraint violation and complementarity $\le 10^{-8}$
  in a single iteration — reporting `Status.OPTIMAL`.
- Acceptable stopping shares the same conditions (plus a relative objective
  change `f_tol`) but requires them to hold for `n_iter` consecutive iterates;
  it follows the IPOPT convention and is **on by default** ($10^{-6}$ for
  15 iterations), reporting `Status.ACCEPTABLE`. Set every tolerance to `None`
  to disable it.
- Optional `max_iter` / `max_time` limits are checked at iteration boundaries and
  report `Status.MAX_ITER` / `Status.MAX_TIME`.
- An IPOPT-style diverging-iterates test reports `Status.UNBOUNDED` when
  $\|x\|_\infty$ exceeds `diverging_iterates_tol` (default $10^{20}$), catching
  an unbounded-below objective before the runaway iterate overflows to a
  misleading numerical failure.
- L-BFGS (compact, Powell-damped) Hessian keeps $N$ positive definite.
