# CrossJIT external C API

This document describes CPython's experimental C API for optimizing across a
Python/native-extension boundary.  The interface is implemented by
[`Include/cpython/jit.h`](../Include/cpython/jit.h) and is not part of the
limited API or stable ABI.

## Scope and design conclusion

The interface is library-neutral.  An extension opts in by registering a
provider for an exact unary operation, binary type pair, or callable.  CPython
discovers operations in a hot Tier 2 trace, builds a small provider-neutral
data-flow graph, and owns the executor and generated-code lifetime.  The
provider owns library semantics, guards, lowering, kernels, and exact fallback.

"Library-neutral" does not mean that CPython can optimize an unmodified
extension or infer arbitrary native semantics.  It means that any opt-in C
extension can use the same protocol without CPython knowing about its object
model.  Tensor, image, dataframe, database, compression, cryptography, and
similar libraries can assign their own dialect and semantic operation IDs.
No NumPy type, dtype, shape, or ufunc rule appears in the ABI or optimizer.

The production optimization unit is the **canonical Region**.  Earlier
operation-specific two-binary fusion was removed because adding a plan kind for
each pattern does not scale beyond one library.  A Region can contain up to 12
nodes and 16 live inputs; each node can consume up to all 16 inputs.  A provider
may advertise lower limits.

The recommended steady-state implementation is:

```text
hot Python trace
    -> exact type/callable provider lookup
    -> atomic plans (semantic anchors and exact replay recipe)
    -> CPython use-def analysis and canonical Region DAG
    -> provider prepare_region()
    -> _EXT_REGION_EXEC
    -> region_fast_eval() for the committed steady state
    -> region_eval() only for cold/PIC/reselection work
```

This path generally has the highest performance potential because it can
remove intermediate Python objects, repeated native dispatch, repeated guards,
and multiple data traversals.  Optimizing only an atomic call reduces fixed
dispatch cost but cannot remove costs between calls.  Provider-supplied machine
code can reduce the remaining external-uop envelope, but it does not replace
Region formation.  The slice-store terminal remains experimental and is not a
required part of the general abstraction.

## ABI status

The current experimental ABI version is `PYJIT_EXT_ABI_VERSION == 12`.
Registration requires an exact ABI match and a complete structure size.  ABI
12 makes the following library-neutral guarantees:

- binary, unary, and callable registration are independent of any extension;
- the Region IR uses dialect IDs, semantic IDs, value IDs, effects, and domains;
- a Region node can represent a call with up to 16 semantic operands;
- a provider that only wants atomic optimization may omit `prepare_region` and
  leave both Region capacity fields zero;
- provider policy is opaque to CPython; diagnostics no longer encode
  library-specific guard or disable reasons;
- the fast Region callback and optional external-code descriptor use the same
  executor ownership and invalidation rules as CPython's native stencils.

Because this is an experimental non-limited API, extensions must be rebuilt
when the ABI version changes.

## Responsibilities

| CPython | Provider |
| --- | --- |
| Per-interpreter registration and lookup | Operation semantics and dialect |
| Tier 2 tracing and symbolic stack analysis | Atomic and Region plan creation |
| Canonical use-def graph construction | Type/layout/property/alias guards |
| Executor, plan table, exits, and invalidation | Kernel selection or generation |
| Stack-reference and result handoff | Exact fallback and Python exceptions |
| Executable allocation and relocation | Optional immutable machine-code bytes |

CPython does not inspect a provider payload and does not assume that an input is
an array.  A provider must not depend on CPython understanding its properties.

## Registration

Providers are registered per interpreter:

```c
int PyJit_RegisterBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg,
    PyJitExtProvider *provider);

int PyJit_RegisterUnaryProvider(
    PyTypeObject *operand_type,
    int unary_opcode,
    PyJitExtProvider *provider);

int PyJit_RegisterCallableProvider(
    PyObject *callable,
    PyJitExtProvider *provider);
```

Each registration has a matching unregister function.  The registry borrows
the provider pointer, so the provider must remain alive until it is unregistered
and all executors containing its plans have gone away.  A static provider is
the simplest valid lifetime strategy.

Registration validates `abi_version`, `struct_size`, `try_plan`, `free_plan`,
and Region capacity.  A Region provider must use the current Core IR version,
advertise 2--12 nodes, and advertise 1--16 live inputs.  An atomic-only provider
may use zero for both capacities.

Until the first provider is registered, CPython retains its ordinary function
entry warm-up threshold.  Successful provider registration enables early entry
probes for subsequently specialized functions, unless the user explicitly set
`PYTHON_JIT_RESUME_INITIAL_VALUE`.  This keeps the unused API off the hot path
while allowing straight-line cross-boundary Regions to be found early.

The binary key is `(exact left type, exact right type, operation)`.  Rich
comparisons use `PYJIT_EXT_BINARY_RICHCOMPARE_BASE + comparison`.  Unary
registration currently exposes negative.  Callable registration is by exact
callable; built-in bound methods can also be found through their C-function
receiver.

## Planning

`try_plan()` receives a temporary `PyJitExtTraceRequest` describing one
operation.  It returns `NULL` to decline or a provider-allocated
`PyJitExtPlan`.  A plan records:

- its kind, provider, operation ID, and provider-owned payload;
- an atomic evaluator or Region evaluator;
- effect flags, operation domain, semantic version, and input count;
- optional fast Region, fallback, store, and code-generation hooks.

The request and all arrays in a Region request are valid only for the duration
of the callback.  A provider that needs them later must copy them.

An atomic plan is both executable and a semantic anchor for Region discovery.
`PYJIT_EXT_PLAN_REGION_ANCHOR_ONLY` tells CPython to restore the ordinary uop if
the plan is not consumed by a Region, avoiding isolated external-dispatch cost.

## Canonical Core IR

CPython constructs a `PyJitExtRegionRequest` only from compatible atomic plans
belonging to the same provider.  It reconstructs live-ins and intermediate
values from the optimized uop stack and emits `PyJitExtRegionNode` records.

Important fields are:

- `dialect_id`: stable namespace chosen by the provider;
- `semantic_id` and `semantic_version`: operation identity inside the dialect;
- `inputs[]`, `input_count`, and `output`: SSA-like data dependencies;
- `effect_flags`: purity, callbacks, mutation, and floating-point policy;
- `domain`: opaque, elementwise, reduction, contraction, or view;
- `graph_hash`: stable hash of graph topology and semantic identities.

Physical call-stack values that are not semantic operands are marked
`PYJIT_EXT_REGION_INPUT_CONTROL`.  For example, a callable or method receiver
may be needed to preserve the physical stack contract but need not be an input
to the provider's mathematical graph.

The builder requires `PYJIT_EXT_EFFECT_NO_PYTHON_CALLBACK` on every Region
node.  Every non-terminal node must also be pure and non-mutating.  These rules
allow CPython to remove intermediate operations without changing observable
Python behavior.  The provider performs all domain-specific checks in
`prepare_region()` and may reject the graph; CPython then retries shorter
prefixes before retaining atomic operations.

The current Core IR is deliberately small.  It forms same-provider,
single-Python-result Regions.  A tuple or other aggregate can be that one
Python result, but CPython does not yet model separately addressable multiple
outputs or cross-provider Regions.

## Runtime result protocol

Evaluator arguments are borrowed references.  A successful result is a new
reference transferred to CPython.

| Result | Meaning |
| --- | --- |
| `PYJIT_EXT_HIT` | The provider completed the operation. |
| `PYJIT_EXT_COLD` | The provider produced the correct result through a cold or replay path; keep the executor. |
| `PYJIT_EXT_MISS` | This speculative executor is no longer suitable; invalidate it and use exact fallback where applicable. |
| `PYJIT_EXT_ERROR` | Propagate the Python exception set by the provider. |

`region_fallback_eval` must implement the exact unfused semantics.  It is used
after a Region `MISS`; it must not silently apply relaxed floating-point,
aliasing, callback, or mutation semantics.

### Fast Region entry

`region_fast_eval(payload, args, nargs)` is the preferred committed hot path:

- non-`NULL` is a successful new-reference result;
- `NULL` with an exception is an error;
- `NULL` without an exception asks CPython to enter `region_eval`.

This compact contract intentionally has no `COLD` or `MISS`.  A cheap guard
failure is a slow-entry event, not an executor invalidation.  The complete
`region_eval` decides whether to replay locally, specialize a new property, or
invalidate the current executor generation.

Providers may implement property signatures and a polymorphic inline cache in
their payload.  The vtable contains optional property/lowering callbacks, but
the current CPython core treats the payload and its cache policy as opaque; the
provider's Region evaluator orchestrates them.

## Optional provider machine code

`provider->get_code(plan, opcode)` may return an immutable `PyJitExtCode` for
an external uop.  Returning `NULL` without an exception selects CPython's
native stencil.  Returning `NULL` with an exception aborts compilation.

The descriptor contains code bytes, optional data bytes, and relocation holes.
CPython validates the ABI, non-empty code, patch kinds, section bounds, trace
targets, allocation arithmetic, and `rel32` range.  It then allocates and marks
the final executor memory executable.  Supported patch values include the
plan, payload, evaluator functions, continuation and error targets, hit/miss
helpers, and an explicit provider-owned external address.

The descriptor and all referenced bytes must remain immutable and valid until
the plan is freed.  This facility is for removing the last generic dispatch
overhead; semantic correctness and the platform-specific calling convention
remain the provider's responsibility.

## Ownership and invalidation

CPython replaces plan pointers embedded in optimized uops with indices into an
executor-owned plan table.  On executor destruction it calls
`provider->free_plan()` once per distinct plan.

When a Region replaces atomic plans, CPython frees the atomic plans unless the
Region declares `PYJIT_EXT_EFFECT_OWNS_ATOMIC_PLANS`.  In that case the provider
must free them with the Region.

`PyJitExtSiteState` outlives an individual executor generation.  It retains
hit/miss counters, disable/retry state, and one provider-owned site cache.  A
provider installing `provider_data` must also install `free_provider_data`.

A runtime `MISS` invalidates and detaches the current executor.  Region disable
uses a bounded backoff lease, so a changing workload can be retraced instead of
being permanently classified by its first failures.  A provider can request a
property-token-based or sticky disable through the site state when that policy
is appropriate.

## Minimal Region provider outline

```c
static PyJitExtPlan *try_plan(
    const PyJitExtTraceRequest *request,
    PyJitExtSiteState *site);
static PyJitExtPlan *prepare_region(
    const PyJitExtRegionRequest *request,
    PyJitExtSiteState *site);
static void free_plan(PyJitExtPlan *plan);

static PyJitExtProvider provider = {
    .abi_version = PYJIT_EXT_ABI_VERSION,
    .struct_size = sizeof(PyJitExtProvider),
    .provider_name = "example",
    .try_plan = try_plan,
    .free_plan = free_plan,
    .prepare_region = prepare_region,
    .dialect_id = UINT64_C(0x6578616d706c6501),
    .dialect_version = 1,
    .core_ir_version = PYJIT_EXT_CORE_IR_VERSION,
    .max_region_nodes = PYJIT_EXT_REGION_MAX_NODES,
    .max_region_inputs = PYJIT_EXT_REGION_MAX_INPUTS,
};
```

Every returned plan must set the current ABI and full structure size.  Region
plans must set `kind`, `provider`, `eval.region_eval`, `region_input_count`, and
the ownership/effect flags required by their fallback strategy.  A provider
should add `region_fast_eval` only after it has a committed entry whose guards
are cheaper than entering the complete Region dispatcher.

## Current limits

- The API is experimental and CPython-specific.
- Registration is exact-type or exact-callable; there is no wildcard or
  provider-priority protocol.
- Canonical Regions contain one provider and one Python-level output.
- The graph is capped at 12 nodes and 16 live/semantic inputs.
- Only operations visible as supported unary, binary, or call uops can anchor a
  Region.
- Python callbacks and intermediate observable mutation are Region barriers.
- The basic-slice store terminal is opt-in and experimental.
- Provider code is trusted native code; structural validation cannot prove its
  semantic or memory safety.

These are bounded implementation limits, not library-specific assumptions.
New operation families should extend the neutral IR or registration mechanism,
not add a library-named plan kind or guard reason.

## Implementation map

| Area | Source |
| --- | --- |
| Public experimental ABI | [`Include/cpython/jit.h`](../Include/cpython/jit.h) |
| Registry, SiteState, Region builder, executor ownership | [`Python/optimizer.c`](../Python/optimizer.c) |
| Atomic plan creation during symbolic optimization | [`Python/optimizer_bytecodes.c`](../Python/optimizer_bytecodes.c) |
| External uop stack and result semantics | [`Python/bytecodes.c`](../Python/bytecodes.c) |
| Provider code descriptors and copy-and-patch | [`Python/jit.c`](../Python/jit.c) |
| Internal runtime declarations | [`Include/internal/pycore_optimizer.h`](../Include/internal/pycore_optimizer.h) |

Generated uop IDs, metadata, optimizer cases, and executor cases must be
regenerated after changing the DSL sources.
