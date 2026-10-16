# CrossJIT Host semantic interface

> Status: Reference / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../crossjit_v3.md).
> If this document conflicts with those files, the normative specification takes precedence.

CrossJIT lets CPython optimize across an extension boundary without importing
the extension's domain model into the optimizer.  The version 3 interface is
organized as four one-way transformations:

```text
D_P -> S_H
(S_H, F_H) -> Pi_H
D_P -> C_P
(Pi_H, C_P) -> Pi_P
```

`D_P` and `C_P` are Provider-private.  CPython owns the generic semantic
vocabulary `S_H`, structural facts `F_H`, and proof `Pi_H`.  A Provider can
recognize operations, project `S_H`, specialize a Host-proven region, lower
guards and an execution plan, and execute those plans.  It cannot select or
expand a Host region, inspect neighboring uops to redefine one, choose a side
exit, or rewrite Host control flow.

## Guarded semantic properties

A projected property may be conditional:

```text
Gamma_i => p_i
```

The Provider enumerates each runtime assumption in `PyJitPrepareResult`.
`PyJitAssumption.semantic_property_mask` identifies the projected properties
that assumption justifies.  A zero mask denotes a condition needed only by
the private specialization contract.  One property may name several
assumptions, and several properties may name the same assumption.
If no assumption names a projected property, that property is unconditional
for the recognized operation and does not require a guard.

The Host validates that every dependency is a region live-in available at the
Host-selected guard point.  It canonicalizes assumptions by Provider-private
`kind`, ordered semantic dependency identities, and `canonical_key`.  When
equivalent descriptions are deduplicated, their semantic-property masks are
unioned.  `payload` is deliberately irrelevant to Host equivalence and remains
Provider-private.

The initial v3 scheduler places the complete deduplicated guard batch at the
region entry.  The Provider implements the checks but cannot move them.  A
guard MISS follows the Host-owned side exit and resumes the original,
unmodified operation sequence.

## Semantic and physical proof

`Pi_H` has two distinct lifetimes:

* The semantic proof class contains Provider identity, normalized `S_H`,
  semantic topology, semantic node/value identity, def-use, closed-DAG,
  escape/materialization, live-in, result, and rewrite-eligibility facts.
  Its collision-checked normalized descriptor excludes observed Python object
  identity, concrete stack positions, and Provider-private contract bytes.
* The physical proof is trace-specific.  CPython independently simulates the
  current uop stack, validates semantic-to-runtime value mappings, proves
  physical liveness, finds concrete input sources and rewrite bounds, and
  records Host-owned deoptimization state.

The current implementation reuses semantic-class identity, not the
computation that proves a member of that class.  Each trace still performs
closed-DAG, escape/materialization, and live-in/result reasoning before its
normalized descriptor is observed.  Consequently the primary counters are
`semantic_class_created` and `semantic_class_reused`; the legacy
`host_proof_created` and `host_proof_reused` test-stat keys are compatibility
aliases and must not be interpreted as saved proof computation.

A semantic-class hit never supplies a physical input mapping.  The
`physical_proof_attempted` and `physical_proof_validated` counters make this
separation observable; a trace with an invalid mapping is rejected even when
its semantic proof class was seen before.

An interior node with `may_call_host` or `may_release_gil` is a Host-visible
effect boundary and cannot be virtualized away.  A result being fresh and
otherwise virtualizable does not weaken that rule.

## Logical and physical arity

The public operation and region representation uses `operands[]` and
`results[]` and therefore remains logically n-ary and multi-result.  The ABI4
physical rewrite implements unary/binary/comparison operations and bounded
calls with positional and keyword mappings, with a single published result.
Limits are 12 nodes, 16 live inputs, and 8 operands per operation. These are
implementation limits of the current executor, not restrictions on the
semantic interface.

## Ownership

CPython owns region formation and boundaries, semantic identities, generic
data/control-flow reasoning, proof and assumption scheduling, guard placement,
side exits, deoptimization, physical rewriting, the executor, and Provider
plan lifetime after publication.  The Provider owns domain recognition,
domain reasoning, semantic projection, domain validation, private
specialization, profitability, guard implementation, plan construction, and
execution.

If `prepare_region()` reports an unsupported tail, CPython may choose and
re-prove a shorter prefix.  The Provider cannot reshape or enlarge it.

## Provider neutrality

Generic Host code must not contain extension-specific dtype, rank, shape,
stride, broadcasting, scalar-category, kernel, layout, element-count, or
profitability knowledge.  Those facts belong in `C_P` and `Pi_P`.

The freeze audit searches the public/internal CrossJIT headers,
`Python/crossjit.c`, the optimizer bridge, bytecode bridge, and interpreter
state for NumPy/domain terms.  The only current `shape` hits are pre-existing
generic optimizer graph/stack diagnostics outside CrossJIT semantics; they do
not encode an extension domain.

After the v3.1 freeze checkpoint, changes to the semantic boundary, Host region
authority, proof ownership, guard-placement ownership, deoptimization model,
or Provider-private boundary require a reproducible generic correctness issue
and a Provider-neutral fix.
