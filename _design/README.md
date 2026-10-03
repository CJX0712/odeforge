# `_design/` -- upstream design documents

This directory holds the **design-phase documents** that preceded the implementation.
They are kept for provenance: the algorithms and invariants described here are the ones
the code in `integrators/`, `routing/`, `autodiff/` and `eval/` implements.

They are **not** part of the runnable deliverable and nothing imports from here.

| file | what it covers |
|---|---|
| `algorithm_kernel.md` | integrator families, composition coefficients, the SympFuse voting rule, differentiable-integration approach |
| `sota_research.md` | the SOTA survey and selection rationale the design was based on |

## How to read this against the code

| design claim | where it landed | status |
|---|---|---|
| Yoshida 4th order as a three-stage composition | `integrators/symplectic.py` | shipped, order verified at 4.003 |
| 6th-order composition for the flagship | `integrators/symplectic.py` | shipped, order verified at 6.000 |
| three-way Hamiltonian routing vote | `routing/sympfuse.py` | shipped |
| "Jacobian of a Hamiltonian system is symmetric" | -- | **corrected**: it is false. The harmonic oscillator's Jacobian is antisymmetric (`A + A^T = 0`). The shipped test is the existence of a symmetric `S` with `A^T S + S A = 0` |
| G2 = 0.5 superiority on trajectory accuracy | `eval/benchmarks.py` | **corrected** to non-inferiority (1.0). 0.5 is unreachable by construction and still fails at 1.0; see `BLOCKED.md` |
| DOP853 as the G3 baseline | `routing/sympfuse.py` | **changed** to explicit SciPy delegation on non-Hamiltonian systems, where a 5th-order own solver is structurally outmatched |

Where the implementation departs from the design, the departure is recorded above and
justified with measurements in `docs/architecture.md` and `BLOCKED.md`.
