/* Host-owned CrossJIT semantic state and per-interpreter provider registry. */
#include "Python.h"
#include "opcode.h"

#include "pycore_crossjit.h"
#include "pycore_interp.h"
#include "pycore_interpframe.h"
#include "pycore_long.h"
#include "pycore_optimizer.h"
#include "pycore_pystate.h"
#include "pycore_uop_metadata.h"

/* Host-owned semantic identity.  Observations are borrowed recorder hints;
 * the monotonically assigned id, rather than JitOptRef bits, defines identity. */
struct PyJitValueRef {
    uint32_t id;                 /* one-based id within this analysis */
    PyObject *observed;           /* borrowed from the recorder */
    PyTypeObject *exact_type;     /* borrowed exact type, or NULL if unknown */
};

/* Provider-facing handle for one Host-owned semantic operation. */
struct PyJitNodeRef {
    uint32_t id;                 /* one-based id within this analysis */
    uint64_t semantic_id;        /* stable Provider operation identity */
    PyJitSemanticSummary summary;/* normalized Host-visible semantics S_H */
    PyJitValueRef *operands[PYJIT_OPERATION_MAX_OPERANDS];
    uint8_t operand_count;
    PyJitValueRef *results[PYJIT_OPERATION_MAX_RESULTS];
    uint8_t result_count;
    PyObject *callable;      /* borrowed recorder hint, pinned only after lowering */
    PyObject *keyword_names;
    PyObject *physical_callable;
    PyObject *receiver;
    uint8_t positional_count;
    uint8_t call_binding_kind;
};

/* Maps a symbolic optimizer value to an independent CrossJIT identity. */
typedef struct {
    JitOptRef symbol;
    PyJitValueRef value;
} CrossJitValueBinding;

/* Internal record for a recognized operation and its trace location. */
typedef struct {
    PyJitNodeRef handle;         /* transient provider-facing view */
    const PyJitProviderSpec *provider; /* borrowed static descriptor */
    uint32_t operands[PYJIT_OPERATION_MAX_OPERANDS];
    uint8_t operand_count;
    uint32_t results[PYJIT_OPERATION_MAX_RESULTS];
    uint8_t result_count;
    uint32_t uop_index;          /* corresponding original trace operation */
} CrossJitNode;

/* Pi_H: extension-independent legality and rewrite proof.  Provider code can
 * observe only the read-only RegionView projected from it. */
typedef struct {
    uint64_t structural_fingerprint;
    uint8_t semantic_class_reused;
    uint8_t node_count;
    uint8_t live_in_count;
    uint8_t result_count;
    uint8_t virtual_value_count;
    uint32_t deopt_uop_index;
    uint32_t rewrite_first;
    uint32_t rewrite_last;
    int16_t input_sources[PYJIT_REGION_MAX_LIVE_INS];
} CrossJitHostRegionProof;

/* A region is a Host-selected contiguous group of semantic nodes. */
typedef struct {
    const PyJitProviderSpec *provider; /* common provider for every node */
    uint32_t first_node;         /* index into analysis state's node vector */
    uint8_t node_count;
    uint8_t input_count;
    PyJitRegionValue inputs[PYJIT_REGION_MAX_LIVE_INS]; /* ordered live-ins */
    uint8_t result_count;
    PyJitRegionValue results[PYJIT_OPERATION_MAX_RESULTS];
    CrossJitHostRegionProof proof;
} CrossJitRegion;

/* Providers only see this read-only wrapper, not internal node storage. */
struct PyJitRegionView {
    const CrossJitRegion *region;       /* borrowed selected candidate */
    const CrossJitAnalysisState *state; /* borrowed semantic arena */
};

/* Lazily allocated per-trace semantic arena.  Each vector is Host-owned and
 * uses the adjacent count/capacity pair; NULL state means zero CrossJIT cost. */
struct CrossJitAnalysisState {
    CrossJitValueBinding *bindings; /* optimizer symbol identities */
    size_t binding_count, binding_capacity;
    CrossJitNode *nodes;         /* recognized semantic operations */
    size_t node_count, node_capacity;
    CrossJitRegion *regions;     /* Host-approved maximal candidates */
    size_t region_count, region_capacity;
    PyJitPrepareDecision last_lowering_decline; /* generic bounded retry reason */
};

#define CROSSJIT_CAPSULE "crossjit.provider"

/* Checks whether a provider supplied a particular optional ABI member. */
static int
provider_has_member(const PyJitProviderSpec *provider, size_t offset,
                    size_t member_size)
{
    return provider->struct_size >= offset + member_size;
}

#define PROVIDER_HAS(SPEC, MEMBER) \
    provider_has_member((SPEC), offsetof(PyJitProviderSpec, MEMBER), \
                        sizeof((SPEC)->MEMBER))

/* Grows a Host-owned vector while preserving its existing contents. */
static int
grow(void **items, size_t item_size, size_t *capacity, size_t needed)
{
    if (needed <= *capacity) {
        return 0;
    }
    size_t cap = *capacity ? *capacity * 2 : 8;
    if (cap < needed || cap > PY_SSIZE_T_MAX / item_size) {
        cap = needed;
    }
    void *new_items = PyMem_Realloc(*items, cap * item_size);
    if (new_items == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    *items = new_items;
    *capacity = cap;
    return 0;
}

/* Allocates the lazy analysis state after the first exact registry hit. */
CrossJitAnalysisState *
_PyCrossJit_AnalysisNew(void)
{
    return PyMem_Calloc(1, sizeof(CrossJitAnalysisState));
}

/* Releases every Host-owned semantic object for one trace. */
void
_PyCrossJit_AnalysisClear(CrossJitAnalysisState *state)
{
    if (state == NULL) {
        return;
    }
    PyMem_Free(state->bindings);
    PyMem_Free(state->nodes);
    PyMem_Free(state->regions);
    memset(state, 0, sizeof(*state));
}

/* Clears and deallocates a per-trace analysis arena. */
void
_PyCrossJit_AnalysisFree(CrossJitAnalysisState *state)
{
    _PyCrossJit_AnalysisClear(state);
    PyMem_Free(state);
}

/* Returns the lazily created exact-key registry for one interpreter. */
static PyObject *
registry_get(PyInterpreterState *interp, int create)
{
    if (interp->crossjit_provider_registry != NULL || !create) {
        return interp->crossjit_provider_registry;
    }
    PyMutex_Lock(&interp->crossjit_provider_registry_mutex);
    if (interp->crossjit_provider_registry == NULL) {
        interp->crossjit_provider_registry = PyDict_New();
    }
    PyMutex_Unlock(&interp->crossjit_provider_registry_mutex);
    return interp->crossjit_provider_registry;
}

/* Builds a value-only dictionary key without retaining provider objects. */
static PyObject *
provider_key(const PyJitProviderKey *key)
{
    PyObject *kind = PyLong_FromLong(key->kind);
    PyObject *a = NULL, *b = NULL, *op = NULL, *result = NULL;
    if (kind == NULL) {
        return NULL;
    }
    if (key->kind == PYJIT_KEY_CALLABLE) {
        if (key->callable == NULL) {
            PyErr_SetString(PyExc_ValueError, "callable provider key has no callable");
            goto done;
        }
        a = PyLong_FromVoidPtr(key->callable);
        if (a != NULL) {
            result = PyTuple_Pack(2, kind, a);
        }
    }
    else {
        if (key->left_type == NULL ||
            ((key->kind == PYJIT_KEY_BINARY ||
              key->kind == PYJIT_KEY_COMPARE) && key->right_type == NULL)) {
            PyErr_SetString(PyExc_ValueError, "type provider key is incomplete");
            goto done;
        }
        a = PyLong_FromVoidPtr(key->left_type);
        b = PyLong_FromVoidPtr(key->right_type);
        op = PyLong_FromLong(key->operation);
        if (a != NULL && b != NULL && op != NULL) {
            result = PyTuple_Pack(4, kind, a, b, op);
        }
    }
done:
    Py_XDECREF(kind);
    Py_XDECREF(a);
    Py_XDECREF(b);
    Py_XDECREF(op);
    return result;
}

/* Checks the ABI boundary before a provider becomes visible to tracing. */
static int
validate_spec(const PyJitProviderSpec *spec)
{
    if (spec == NULL || spec->abi_version != PYJIT_PROVIDER_ABI_VERSION ||
        !PROVIDER_HAS(spec, destroy_exec_plan)) {
        PyErr_SetString(PyExc_ValueError, "unsupported CrossJIT provider ABI");
        return -1;
    }
    if (spec->recognize == NULL || spec->project_semantics == NULL ||
        spec->prepare_region == NULL || spec->lower_region == NULL ||
        spec->run_region == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "CrossJIT provider is missing a required v3 callback");
        return -1;
    }
    if (spec->key_count && spec->keys == NULL) {
        PyErr_SetString(PyExc_ValueError, "CrossJIT provider has no key array");
        return -1;
    }
    return 0;
}

/* Registers every exact key atomically with respect to interpreter access. */
PyAPI_FUNC(int)
PyJit_RegisterProvider(const PyJitProviderSpec *spec)
{
    if (validate_spec(spec) < 0) {
        return -1;
    }
    PyThreadState *tstate = _PyThreadState_GET();
    if (tstate == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no current thread state");
        return -1;
    }
    PyObject *registry = registry_get(tstate->interp, 1);
    if (registry == NULL) {
        return -1;
    }
    for (size_t i = 0; i < spec->key_count; i++) {
        PyObject *key = provider_key(&spec->keys[i]);
        if (key == NULL) {
            return -1;
        }
        PyObject *item = PyDict_GetItemWithError(registry, key);
        if (item != NULL) {
            const PyJitProviderSpec *old = PyCapsule_GetPointer(item, CROSSJIT_CAPSULE);
            Py_DECREF(key);
            if (old == spec) {
                continue;
            }
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "conflicting CrossJIT provider key");
            }
            return -1;
        }
        if (PyErr_Occurred()) {
            Py_DECREF(key);
            return -1;
        }
        PyObject *capsule = PyCapsule_New((void *)spec, CROSSJIT_CAPSULE, NULL);
        int rc = capsule == NULL ? -1 : PyDict_SetItem(registry, key, capsule);
        Py_XDECREF(capsule);
        Py_DECREF(key);
        if (rc < 0) {
            return -1;
        }
    }
    return 0;
}

/* Removes only keys that still map to this exact provider descriptor. */
PyAPI_FUNC(int)
PyJit_UnregisterProvider(const PyJitProviderSpec *spec)
{
    if (validate_spec(spec) < 0) {
        return -1;
    }
    PyThreadState *tstate = _PyThreadState_GET();
    if (tstate == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no current thread state");
        return -1;
    }
    PyObject *registry = registry_get(tstate->interp, 0);
    if (registry == NULL) {
        return 0;
    }
    for (size_t i = 0; i < spec->key_count; i++) {
        PyObject *key = provider_key(&spec->keys[i]);
        if (key == NULL) {
            return -1;
        }
        PyObject *item = PyDict_GetItemWithError(registry, key);
        if (item != NULL && PyCapsule_GetPointer(item, CROSSJIT_CAPSULE) == spec) {
            if (PyDict_DelItem(registry, key) < 0) {
                Py_DECREF(key);
                return -1;
            }
        }
        else if (PyErr_Occurred()) {
            Py_DECREF(key);
            return -1;
        }
        Py_DECREF(key);
    }
    return 0;
}

/* Performs one exact lookup; it never scans the provider population. */
static const PyJitProviderSpec *
lookup(PyInterpreterState *interp, const PyJitProviderKey *key)
{
    PyObject *registry = registry_get(interp, 0);
    if (registry == NULL) {
        return NULL;
    }
    PyObject *dict_key = provider_key(key);
    if (dict_key == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyObject *item = PyDict_GetItemWithError(registry, dict_key);
    Py_DECREF(dict_key);
    if (item == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const PyJitProviderSpec *spec = PyCapsule_GetPointer(item, CROSSJIT_CAPSULE);
    if (spec == NULL) {
        PyErr_Clear();
    }
    return spec;
}

/* Returns whether semantic analysis is enabled for this process. */
int
_PyCrossJit_IsEnabled(void)
{
    const char *value = Py_GETENV("PYTHON_JIT_CROSSJIT");
    return value != NULL &&
           (strcmp(value, "1") == 0 || strcmp(value, "semantics") == 0);
}

/* Returns whether analysis should run while region lowering stays disabled. */
int
_PyCrossJit_IsSemanticsOnly(void)
{
    const char *value = Py_GETENV("PYTHON_JIT_CROSSJIT");
    return value != NULL && strcmp(value, "semantics") == 0;
}

/* Finds the exact unary operand type and operation registration. */
const PyJitProviderSpec *
_PyCrossJit_LookupUnaryProvider(PyInterpreterState *interp,
                                PyTypeObject *operand, int operation)
{
    PyJitProviderKey key = {PYJIT_KEY_UNARY, operation, operand, NULL, NULL};
    return operand != NULL ? lookup(interp, &key) : NULL;
}

/* Finds the exact comparison type-pair and operation registration. */
const PyJitProviderSpec *
_PyCrossJit_LookupCompareProvider(PyInterpreterState *interp,
                                  PyTypeObject *left, PyTypeObject *right,
                                  int operation)
{
    PyJitProviderKey key = {PYJIT_KEY_COMPARE, operation, left, right, NULL};
    return left != NULL && right != NULL ? lookup(interp, &key) : NULL;
}

static int
crossjit_callable_equal(PyObject *left, PyObject *right)
{
    if (left==right) return 1;
    return left && right && PyCFunction_CheckExact(left) && PyCFunction_CheckExact(right) &&
        PyCFunction_GET_FUNCTION(left)==PyCFunction_GET_FUNCTION(right) &&
        PyCFunction_GET_SELF(left)==PyCFunction_GET_SELF(right) &&
        PyCFunction_GET_FLAGS(left)==PyCFunction_GET_FLAGS(right);
}

/* Finds a registration for the exact callable object identity. */
const PyJitProviderSpec *
_PyCrossJit_LookupCallableProvider(PyInterpreterState *interp, PyObject *callable)
{
    PyJitProviderKey key = {PYJIT_KEY_CALLABLE, 0, NULL, NULL, callable};
    const PyJitProviderSpec *found=callable != NULL ? lookup(interp, &key) : NULL;
    if (found || !callable || !PyCFunction_CheckExact(callable)) return found;
    PyObject *registry=registry_get(interp,0);
    if (!registry) return NULL;
    Py_ssize_t pos=0;
    PyObject *entry_key,*capsule;
    while (PyDict_Next(registry,&pos,&entry_key,&capsule)) {
        const PyJitProviderSpec *spec=PyCapsule_GetPointer(capsule,CROSSJIT_CAPSULE);
        if (!spec) return NULL;
        for (size_t i=0;i<spec->key_count;i++) {
            if (spec->keys[i].kind==PYJIT_KEY_CALLABLE &&
                    crossjit_callable_equal(callable,spec->keys[i].callable)) return spec;
        }
    }
    return NULL;
}

/* Match a method descriptor and its exact receiver to a registered bound
 * builtin, without attribute lookup or invoking descriptor/user code. */
static PyObject *
crossjit_registered_method(PyObject *descriptor, PyObject *receiver)
{
    if (!descriptor || !receiver || !Py_IS_TYPE(descriptor,&PyMethodDescr_Type)) return NULL;
    PyMethodDescrObject *method=(PyMethodDescrObject *)descriptor;
    if (Py_TYPE(receiver)!=method->d_common.d_type) return NULL;
    PyObject *registry=registry_get(_PyInterpreterState_GET(),0);
    if (!registry) return NULL;
    Py_ssize_t pos=0;
    PyObject *key,*capsule;
    while (PyDict_Next(registry,&pos,&key,&capsule)) {
        const PyJitProviderSpec *spec=PyCapsule_GetPointer(capsule,CROSSJIT_CAPSULE);
        if (!spec) return NULL;
        for (size_t i=0;i<spec->key_count;i++) {
            PyObject *bound=spec->keys[i].callable;
            if (spec->keys[i].kind==PYJIT_KEY_CALLABLE && bound && PyCFunction_CheckExact(bound) &&
                PyCFunction_GET_SELF(bound)==receiver &&
                PyCFunction_GET_FUNCTION(bound)==method->d_method->ml_meth &&
                PyCFunction_GET_FLAGS(bound)==method->d_method->ml_flags) return bound;
        }
    }
    return NULL;
}

int
_PyCrossJit_ShouldObserveCallSite(PyObject *callable, PyObject *receiver)
{
    if (!_PyCrossJit_IsEnabled()) return 0;
    return receiver ? crossjit_registered_method(callable,receiver)!=NULL :
        _PyCrossJit_LookupCallableProvider(_PyInterpreterState_GET(),callable)!=NULL;
}

/* Retains comparison objects only when the exact type-pair is registered. */
int
_PyCrossJit_ShouldObserveCompare(PyObject *left, PyObject *right,
                                 int operation)
{
    return _PyCrossJit_IsEnabled() && left != NULL && right != NULL &&
           _PyCrossJit_LookupCompareProvider(
               _PyInterpreterState_GET(), Py_TYPE(left), Py_TYPE(right),
               operation) != NULL;
}

/* Drops the per-interpreter registry during interpreter finalization. */
void
_PyCrossJit_InterpreterClear(PyInterpreterState *interp)
{
    Py_CLEAR(interp->crossjit_provider_registry);
    Py_CLEAR(interp->crossjit_host_proof_cache);
}

/* Returns a stable, test-facing snapshot of the current interpreter counters. */
PyAPI_FUNC(PyObject *)
_PyCrossJit_GetStats(PyInterpreterState *interp)
{
    PyObject *stats = PyDict_New();
    if (stats == NULL) {
        return NULL;
    }
#define ADD_STAT(name) do {                                                \
        PyObject *value = PyLong_FromUnsignedLongLong(                     \
            interp->crossjit_stats.name);                                  \
        if (value == NULL || PyDict_SetItemString(stats, #name, value) < 0) { \
            Py_XDECREF(value);                                             \
            Py_DECREF(stats);                                              \
            return NULL;                                                   \
        }                                                                  \
        Py_DECREF(value);                                                  \
    } while (0)
    ADD_STAT(recognized_nodes);
    ADD_STAT(provider_first_recognized);
    ADD_STAT(scalar_speculations_avoided);
    ADD_STAT(assumptions_before_dedup);
    ADD_STAT(assumptions_after_dedup);
    ADD_STAT(host_regions_considered);
    ADD_STAT(host_regions_proven);
    ADD_STAT(semantic_class_created);
    ADD_STAT(semantic_class_reused);
    ADD_STAT(semantic_class_hit);
    ADD_STAT(semantic_class_miss);
    ADD_STAT(physical_proof_attempted);
    ADD_STAT(physical_proof_validated);
    ADD_STAT(provider_specialization_created);
    ADD_STAT(provider_specialization_rebuilt);
    ADD_STAT(reject_unsupported_node);
    ADD_STAT(reject_unsupported_contract);
    ADD_STAT(reject_unprofitable);
    ADD_STAT(materialized_values);
    ADD_STAT(virtual_values);
    ADD_STAT(transient_local_regions);
    ADD_STAT(transient_local_exec_hits);
    ADD_STAT(transient_local_guard_misses);
    ADD_STAT(type_guard_regions);
    ADD_STAT(type_guard_exec_hits);
    ADD_STAT(type_guard_checks);
    ADD_STAT(type_guard_misses);
    ADD_STAT(rejected_regions);
    ADD_STAT(formed_regions);
    ADD_STAT(rewritten_regions);
    ADD_STAT(published_regions);
    ADD_STAT(guard_hits);
    ADD_STAT(combined_guard_exec_calls);
    ADD_STAT(guard_misses);
    ADD_STAT(exec_hits);
    ADD_STAT(exec_misses);
#undef ADD_STAT
    /* Temporary compatibility aliases for existing Provider tests.  These
     * counters describe exact semantic-class observations, not avoided Host
     * proof computation. */
    PyObject *created = PyLong_FromUnsignedLongLong(
        interp->crossjit_stats.semantic_class_created);
    PyObject *reused = PyLong_FromUnsignedLongLong(
        interp->crossjit_stats.semantic_class_reused);
    if (created == NULL || reused == NULL ||
        PyDict_SetItemString(stats, "host_proof_created", created) < 0 ||
        PyDict_SetItemString(stats, "host_proof_reused", reused) < 0) {
        Py_XDECREF(created);
        Py_XDECREF(reused);
        Py_DECREF(stats);
        return NULL;
    }
    Py_DECREF(created);
    Py_DECREF(reused);
    return stats;
}

/* Resets only CrossJIT experiment counters for deterministic test isolation. */
PyAPI_FUNC(void)
_PyCrossJit_ResetStats(PyInterpreterState *interp)
{
    memset(&interp->crossjit_stats, 0, sizeof(interp->crossjit_stats));
    if (interp->crossjit_host_proof_cache != NULL) {
        PyDict_Clear(interp->crossjit_host_proof_cache);
    }
}

/* Returns a checked executor slot without exposing its storage to providers. */
static CrossJitExecutorPlan *
executor_plan(_PyExecutorObject *executor, uint32_t plan_index,
              Py_ssize_t input_count)
{
    if (executor == NULL || plan_index >= executor->crossjit_plan_count) {
        return NULL;
    }
    CrossJitExecutorPlan *plan = &executor->crossjit_plans[plan_index];
    if (plan->provider == NULL ||
        (input_count >= 0 && plan->input_count != input_count)) {
        return NULL;
    }
    return plan;
}

/* Materializes the provider input vector from stack values and frame locals. */
int
_PyCrossJit_ExecutorPrepareInputs(
    _PyExecutorObject *executor, uint32_t plan_index,
    _PyInterpreterFrame *frame, _PyStackRef *stack_inputs,
    Py_ssize_t stack_input_count, PyObject **inputs, Py_ssize_t capacity)
{
    CrossJitExecutorPlan *plan = executor_plan(executor, plan_index, -1);
    if (plan == NULL || frame == NULL || inputs == NULL ||
        stack_input_count != plan->stack_input_count ||
        capacity < plan->input_count) {
        return -1;
    }
    _PyStackRef *locals = _PyFrame_GetLocalsArray(frame);
    for (uint8_t i=0;i<plan->call_guard_count;i++) {
        int source=plan->call_guard_sources[i];
        _PyStackRef value=source>=0?stack_inputs[source]:locals[-source-1];
        PyObject *expected=plan->call_guard_objects[i];
        if (expected ? (PyStackRef_IsNull(value) ||
                !crossjit_callable_equal(PyStackRef_AsPyObjectBorrow(value),expected)) : !PyStackRef_IsNull(value)) {
            _PyInterpreterState_GET()->crossjit_stats.guard_misses++;
            return -1;
        }
    }
    /* A suppressed temporary store must neither release a previous object
     * nor leave a virtual value in locals at exit.  The physical proof checks
     * the latter; this entry guard checks the former without consuming anything.
     * It is Host-owned and precedes every Provider guard/execution callback. */
    for (uint8_t i = 0; i < plan->empty_local_count; i++) {
        if (!PyStackRef_IsNull(locals[plan->empty_locals[i]])) {
            _PyInterpreterState_GET()->crossjit_stats.guard_misses++;
            _PyInterpreterState_GET()->crossjit_stats.transient_local_guard_misses++;
            return -1;
        }
    }
    for (uint8_t i = 0; i < plan->input_count; i++) {
        int source = plan->input_sources[i];
        if (source == INT16_MIN) {
            if (plan->input_constants[i] == NULL) {
                return -1;
            }
            inputs[i] = plan->input_constants[i];
            continue;
        }
        _PyStackRef ref;
        if (source >= 0) {
            if (source >= stack_input_count) {
                return -1;
            }
            ref = stack_inputs[source];
        }
        else {
            ref = locals[-source - 1];
        }
        if (PyStackRef_IsNull(ref)) {
            return -1;
        }
        inputs[i] = PyStackRef_AsPyObjectBorrow(ref);
    }
    /* Replays the original structural type tests at the common entry. The
     * physical proof independently maps each tested value to a live-in; a
     * failure leaves both stack and locals untouched for first-op replay. */
    for (uint8_t i = 0; i < plan->type_guard_count; i++) {
        _PyInterpreterState_GET()->crossjit_stats.type_guard_checks++;
        if (Py_TYPE(inputs[plan->type_guard_inputs[i]]) !=
                plan->type_guard_types[i]) {
            _PyInterpreterState_GET()->crossjit_stats.guard_misses++;
            _PyInterpreterState_GET()->crossjit_stats.type_guard_misses++;
            return -1;
        }
    }
    return plan->input_count;
}

/* Converts an invalid provider result into a normal CPython error result. */
static PyJitResult
provider_protocol_error(const char *message)
{
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, message);
    }
    return PYJIT_RESULT_ERROR;
}

/* Exposes owned Host constants; Provider plans remain opaque to this hook. */
int
_PyCrossJit_ExecutorPlansTraverse(_PyExecutorObject *executor,
                                 visitproc visit, void *arg)
{
    for (uint16_t i = 0; i < executor->crossjit_plan_count; i++) {
        CrossJitExecutorPlan *plan = &executor->crossjit_plans[i];
        for (uint8_t j=0;j<plan->call_guard_count;j++) { Py_VISIT(plan->call_guard_objects[j]); }
        for (uint8_t input = 0; input < plan->input_count; input++) {
            Py_VISIT(plan->input_constants[input]);
        }
    }
    return 0;
}

/* Detaches first so recursive finalization cannot destroy a plan twice. */
void
_PyCrossJit_ExecutorPlansClear(_PyExecutorObject *executor)
{
    if (executor == NULL || executor->crossjit_plans == NULL) {
        return;
    }
    CrossJitExecutorPlan *plans = executor->crossjit_plans;
    uint16_t plan_count = executor->crossjit_plan_count;
    executor->crossjit_plans = NULL;
    executor->crossjit_plan_count = 0;
    for (uint16_t i = 0; i < plan_count; i++) {
        CrossJitExecutorPlan *plan = &plans[i];
        if (plan->guard_plan != NULL && plan->provider != NULL &&
            PROVIDER_HAS(plan->provider, destroy_guard_plan) &&
            plan->provider->destroy_guard_plan != NULL) {
            plan->provider->destroy_guard_plan(plan->guard_plan);
        }
        if (plan->exec_plan != NULL && plan->provider != NULL &&
            PROVIDER_HAS(plan->provider, destroy_exec_plan) &&
            plan->provider->destroy_exec_plan != NULL) {
            plan->provider->destroy_exec_plan(plan->exec_plan);
        }
        for (uint8_t input = 0; input < plan->input_count; input++) {
            Py_CLEAR(plan->input_constants[input]);
        }
        for (uint8_t j=0;j<plan->call_guard_count;j++) { Py_CLEAR(plan->call_guard_objects[j]); }
        plan->call_guard_count=0;
    }
    PyMem_Free(plans);
}

PyJitResult
_PyCrossJit_ExecutorRunGuards(_PyExecutorObject *executor, uint32_t plan_index,
                              PyObject *const *inputs, Py_ssize_t input_count)
{
    /* Runs guard plans before execution while every stack input is still live. */
    CrossJitExecutorPlan *plan = executor_plan(executor, plan_index, input_count);
    if (plan == NULL) {
        return PYJIT_RESULT_MISS;
    }
    if (plan->guard_plan == NULL) {
        return PYJIT_RESULT_HIT;
    }
    if (!PROVIDER_HAS(plan->provider, run_guards) ||
        plan->provider->run_guards == NULL) {
        return PYJIT_RESULT_MISS;
    }
    PyJitResult status = plan->provider->run_guards(
        plan->guard_plan, inputs, input_count);
    if (status == PYJIT_RESULT_HIT) {
        _PyInterpreterState_GET()->crossjit_stats.guard_hits++;
    }
    else if (status == PYJIT_RESULT_MISS) {
        _PyInterpreterState_GET()->crossjit_stats.guard_misses++;
    }
    if (status == PYJIT_RESULT_HIT || status == PYJIT_RESULT_MISS) {
        return status;
    }
    if (status == PYJIT_RESULT_ERROR) {
        return provider_protocol_error("CrossJIT guard plan failed without an exception");
    }
    return provider_protocol_error("CrossJIT guard plan returned an invalid status");
}

PyJitResult
_PyCrossJit_ExecutorRunRegion(_PyExecutorObject *executor, uint32_t plan_index,
                              PyObject *const *inputs, Py_ssize_t input_count,
                              PyObject **result)
{
    /* Runs one provider plan and accepts a result only after a confirmed HIT. */
    if (result == NULL) {
        return provider_protocol_error("CrossJIT region call has no result slot");
    }
    *result = NULL;
    CrossJitExecutorPlan *plan = executor_plan(executor, plan_index, input_count);
    if (plan == NULL || plan->exec_plan == NULL ||
        !PROVIDER_HAS(plan->provider, run_region) ||
        plan->provider->run_region == NULL) {
        return PYJIT_RESULT_MISS;
    }
    PyJitResult status;
    if (plan->combined_guards) {
        if (!PROVIDER_HAS(plan->provider, run_guarded_region) ||
            plan->provider->run_guarded_region == NULL) {
            return PYJIT_RESULT_MISS;
        }
        PyJitResult guard_status = PYJIT_RESULT_ERROR;
        _PyInterpreterState_GET()->crossjit_stats.combined_guard_exec_calls++;
        status = plan->provider->run_guarded_region(
            plan->guard_plan, plan->exec_plan, inputs, input_count,
            &guard_status, result);
        if (guard_status == PYJIT_RESULT_HIT) {
            _PyInterpreterState_GET()->crossjit_stats.guard_hits++;
        }
        else if (guard_status == PYJIT_RESULT_MISS) {
            _PyInterpreterState_GET()->crossjit_stats.guard_misses++;
        }
        else if (guard_status != PYJIT_RESULT_ERROR) {
            Py_XDECREF(*result);
            *result = NULL;
            return provider_protocol_error("CrossJIT combined guard returned invalid status");
        }
        if (guard_status != PYJIT_RESULT_HIT) {
            if (*result != NULL || status != guard_status) {
                Py_XDECREF(*result);
                *result = NULL;
                return provider_protocol_error("CrossJIT execution proceeded after failed guards");
            }
            return guard_status == PYJIT_RESULT_MISS ? guard_status :
                provider_protocol_error("CrossJIT combined guard failed without an exception");
        }
    }
    else {
        status = plan->provider->run_region(
            plan->exec_plan, inputs, input_count, result);
    }
    if (status == PYJIT_RESULT_HIT) {
        _PyInterpreterState_GET()->crossjit_stats.exec_hits++;
        if (plan->empty_local_count) {
            _PyInterpreterState_GET()->crossjit_stats.transient_local_exec_hits++;
        }
        if (plan->type_guard_count) {
            _PyInterpreterState_GET()->crossjit_stats.type_guard_exec_hits++;
        }
    }
    else if (status == PYJIT_RESULT_MISS) {
        _PyInterpreterState_GET()->crossjit_stats.exec_misses++;
    }
    if (status == PYJIT_RESULT_HIT) {
        if (*result != NULL) {
            return status;
        }
        return provider_protocol_error("CrossJIT region HIT has no result");
    }
    if (*result != NULL) {
        Py_DECREF(*result);
        *result = NULL;
        return provider_protocol_error("CrossJIT region returned a result on MISS or ERROR");
    }
    if (status == PYJIT_RESULT_MISS) {
        return status;
    }
    if (status == PYJIT_RESULT_ERROR) {
        return provider_protocol_error("CrossJIT region plan failed without an exception");
    }
    return provider_protocol_error("CrossJIT region plan returned an invalid status");
}

/* Creates or retrieves an identity independent of JitOptRef's bit layout. */
static PyJitValueRef *
value_for_symbol(CrossJitAnalysisState *state, JitOptContext *ctx, JitOptRef symbol)
{
    for (size_t i = 0; i < state->binding_count; i++) {
        /* Borrowed/unique tags describe stack ownership, not value identity. */
        if (PyJitRef_Unwrap(state->bindings[i].symbol) ==
            PyJitRef_Unwrap(symbol)) {
            PyJitValueRef *value = &state->bindings[i].value;
            /* A producer can create the identity before a later recorder has
             * observed its materialized result.  Refresh metadata without
             * changing the Host-owned identity. */
            if (value->observed == NULL) {
                value->observed = _Py_uop_sym_get_probable_value(symbol);
            }
            if (value->exact_type == NULL) {
                value->exact_type = _Py_uop_sym_get_probable_type(symbol);
            }
            return value;
        }
    }
    CrossJitValueBinding *old_bindings = state->bindings;
    if (state->binding_count == UINT32_MAX ||
        grow((void **)&state->bindings, sizeof(*state->bindings),
             &state->binding_capacity, state->binding_count + 1) < 0) {
        return NULL;
    }
    if (state->bindings != old_bindings) {
        /* Node handles expose pointers into the Host value arena.  Rebuild
         * them from stable value ids whenever the backing vector moves. */
        for (size_t i = 0; i < state->node_count; i++) {
            CrossJitNode *node = &state->nodes[i];
            for (uint8_t operand = 0; operand < node->operand_count; operand++) {
                uint32_t id = node->operands[operand];
                assert(id > 0 && id <= state->binding_count);
                node->handle.operands[operand] =
                    &state->bindings[id - 1].value;
            }
            for (uint8_t result = 0; result < node->result_count; result++) {
                uint32_t id = node->results[result];
                assert(id > 0 && id <= state->binding_count);
                node->handle.results[result] = &state->bindings[id - 1].value;
            }
        }
    }
    PyObject *observed = _Py_uop_sym_get_probable_value(symbol);
    PyTypeObject *type = _Py_uop_sym_get_type(symbol);
    if (type == NULL && observed != NULL) {
        type = Py_TYPE(observed);
    }
    CrossJitValueBinding *binding = &state->bindings[state->binding_count];
    binding->symbol = symbol;
    binding->value = (PyJitValueRef){(uint32_t)(state->binding_count + 1), observed, type};
    state->binding_count++;
    return &binding->value;
}

/* Validates and normalizes one Provider projection before it becomes S_H. */
static int
normalize_semantic_summary(PyJitSemanticSummary *summary)
{
    if (summary->struct_size != sizeof(*summary) ||
        summary->mutation < PYJIT_MUTATION_NONE ||
        summary->mutation > PYJIT_MUTATION_UNKNOWN ||
        summary->result_alias < PYJIT_ALIAS_FRESH ||
        summary->result_alias > PYJIT_ALIAS_UNKNOWN ||
        summary->materialization < PYJIT_MATERIALIZATION_VIRTUALIZABLE ||
        summary->materialization > PYJIT_MATERIALIZATION_UNKNOWN ||
        summary->replayability < PYJIT_REPLAYABLE ||
        summary->replayability > PYJIT_REPLAYABILITY_UNKNOWN ||
        summary->reserved8 != 0 || summary->reserved[0] != 0 ||
        summary->reserved[1] != 0 ||
        (summary->exact_result_type != NULL &&
         !PyType_Check((PyObject *)summary->exact_result_type))) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid CrossJIT semantic summary");
        return -1;
    }
    summary->may_raise = !!summary->may_raise;
    summary->may_call_host = !!summary->may_call_host;
    summary->may_release_gil = !!summary->may_release_gil;
    return 0;
}

/* Recognition and projection are deliberately cheap: no Provider contract,
 * assumptions, profitability policy, or kernel choice is constructed here. */
static int
recognize_operation(JitOptContext *ctx, const PyJitProviderSpec *provider,
                  PyJitOperationKind kind, int operation, PyObject *callable, PyObject *keyword_names,
                  JitOptRef *operand_symbols, uint8_t operand_count,
                  CrossJitRecognition *prepared)
{
    CrossJitAnalysisState *state = ctx->crossjit;
    bool new_state = false;
    if (state == NULL) {
        state = _PyCrossJit_AnalysisNew();
        if (state == NULL) {
            return PyErr_NoMemory(), -1;
        }
        new_state = true;
    }

    PyJitValueRef *operands[PYJIT_OPERATION_MAX_OPERANDS] = {NULL};
    uint32_t operand_ids[PYJIT_OPERATION_MAX_OPERANDS] = {0};
    for (uint8_t i = 0; i < operand_count; i++) {
        operands[i] = value_for_symbol(state, ctx, operand_symbols[i]);
        if (operands[i] == NULL) {
            if (new_state) {
                _PyCrossJit_AnalysisFree(state);
            }
            return -1;
        }
        operand_ids[i] = operands[i]->id;
    }
    /* Any operand insertion can grow the arena. Refresh borrowed handles
     * from stable identities before calling the Provider. */
    for (uint8_t i = 0; i < operand_count; i++) {
        assert(operand_ids[i] > 0 && operand_ids[i] <= state->binding_count);
        operands[i] = &state->bindings[operand_ids[i] - 1].value;
    }

    PyJitRecognizeRequest request = {
        .kind = kind,
        .operation = operation,
        .callable = callable,
        .operands = operands,
        .operand_count = operand_count,
        .result_count = 1,
        .has_keywords = keyword_names != NULL,
        .keyword_names = keyword_names,
        .call_descriptor_version = kind==PYJIT_OPERATION_CALL ? PYJIT_CALL_DESCRIPTOR_VERSION : 0,
        .positional_count = operand_count-(keyword_names?(uint8_t)PyTuple_GET_SIZE(keyword_names):0),
    };
    PyJitRecognition recognition = {0};
    PyJitResult status = provider->recognize(&request, &recognition);
    if (status == PYJIT_RESULT_MISS) {
        if (new_state) {
            _PyCrossJit_AnalysisFree(state);
        }
        return 0;
    }
    if (status != PYJIT_RESULT_HIT || recognition.semantic_id == 0) {
        if (new_state) {
            _PyCrossJit_AnalysisFree(state);
        }
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_SystemError,
                            "CrossJIT recognize returned an invalid result");
        }
        return -1;
    }

    PyJitSemanticSummary summary = {
        .struct_size = sizeof(PyJitSemanticSummary),
    };
    status = provider->project_semantics(&request, &recognition, &summary);
    if (status == PYJIT_RESULT_MISS) {
        if (new_state) {
            _PyCrossJit_AnalysisFree(state);
        }
        return 0;
    }
    if (status != PYJIT_RESULT_HIT ||
        normalize_semantic_summary(&summary) < 0) {
        if (new_state) {
            _PyCrossJit_AnalysisFree(state);
        }
        if (status == PYJIT_RESULT_ERROR && !PyErr_Occurred()) {
            PyErr_SetString(PyExc_SystemError,
                            "CrossJIT semantic projection failed");
        }
        return -1;
    }

    *prepared = (CrossJitRecognition){provider, recognition, summary};
    ctx->crossjit = state;
    return 1;
}

static int
bind_operation(JitOptContext *ctx, JitOptRef *operand_symbols,
               uint8_t operand_count, JitOptRef result, uint32_t uop_index,
               const CrossJitRecognition *prepared)
{
    CrossJitAnalysisState *state = ctx->crossjit;
    assert(state != NULL);
    uint32_t operand_ids[PYJIT_OPERATION_MAX_OPERANDS];
    for (uint8_t i = 0; i < operand_count; i++) {
        PyJitValueRef *value = value_for_symbol(state, ctx, operand_symbols[i]);
        if (value == NULL) {
            return -1;
        }
        operand_ids[i] = value->id;
    }
    PyJitValueRef *res = value_for_symbol(state, ctx, result);
    if (res == NULL) {
        return -1;
    }
    PyJitValueRef *operands[PYJIT_OPERATION_MAX_OPERANDS];
    for (uint8_t i = 0; i < operand_count; i++) {
        operands[i] = &state->bindings[operand_ids[i] - 1].value;
    }
    if (state->node_count == UINT32_MAX ||
        grow((void **)&state->nodes, sizeof(*state->nodes),
             &state->node_capacity, state->node_count + 1) < 0) {
        return -1;
    }
    CrossJitNode *node = &state->nodes[state->node_count];
    memset(node, 0, sizeof(*node));
    node->provider = prepared->provider;
    node->operand_count = operand_count;
    node->result_count = 1;
    node->uop_index = uop_index;
    node->handle.id = (uint32_t)(state->node_count + 1);
    node->handle.semantic_id = prepared->recognition.semantic_id;
    node->handle.summary = prepared->summary;
    node->handle.operand_count = operand_count;
    node->handle.result_count = 1;
    for (uint8_t i = 0; i < operand_count; i++) {
        node->operands[i] = operands[i]->id;
        node->handle.operands[i] = operands[i];
    }
    node->results[0] = res->id;
    node->handle.results[0] = res;
    res->exact_type = prepared->summary.exact_result_type;
    state->node_count++;
    _PyInterpreterState_GET()->crossjit_stats.recognized_nodes++;
    /* Provider projections can depend on assumptions supplied only after
     * region preparation. Keep their type facts in the CrossJIT value arena:
     * publishing them to the ordinary optimizer here would narrow observers
     * even when preparation declines or a later guard replays the operation.
     * The physical rewrite and its scheduled guards establish specialization
     * without changing the original operation's unconditional result type. */
    ctx->crossjit = state;
    return 1;
}

static int
analyze_operation(JitOptContext *ctx, const PyJitProviderSpec *provider,
                  PyJitOperationKind kind, int operation, PyObject *callable,
                  JitOptRef *operands, uint8_t operand_count,
                  JitOptRef result, uint32_t uop_index)
{
    CrossJitRecognition prepared;
    int status = recognize_operation(ctx, provider, kind, operation, callable, NULL,
                                     operands, operand_count, &prepared);
    if (status <= 0) {
        return status;
    }
    return bind_operation(ctx, operands, operand_count, result, uop_index,
                          &prepared);
}

/* Conditional producer facts can guide another Provider recognition without
 * escaping into the ordinary optimizer. Admission still requires a prepared
 * contract and Host-scheduled guards before any operation is replaced. */
static PyTypeObject *
crossjit_recognition_type(JitOptContext *ctx, JitOptRef symbol)
{
    CrossJitAnalysisState *state = ctx->crossjit;
    if (state != NULL) {
        for (size_t i = 0; i < state->binding_count; i++) {
            if (PyJitRef_Unwrap(state->bindings[i].symbol) ==
                    PyJitRef_Unwrap(symbol) &&
                    state->bindings[i].value.exact_type != NULL) {
                return state->bindings[i].value.exact_type;
            }
        }
    }
    return _Py_uop_sym_get_probable_type(symbol);
}

static const PyJitProviderSpec *
lookup_type_pair(JitOptContext *ctx, JitOptRef left, JitOptRef right,
                 PyJitProviderKeyKind key_kind, int operation)
{
    PyTypeObject *left_type = crossjit_recognition_type(ctx, left);
    PyTypeObject *right_type = crossjit_recognition_type(ctx, right);
    if (left_type == NULL || right_type == NULL) {
        return NULL;
    }
    PyJitProviderKey key = {
        key_kind, operation, left_type, right_type, NULL
    };
    const PyJitProviderSpec *provider = lookup(_PyInterpreterState_GET(), &key);
    if (provider == NULL && Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
        fprintf(stderr, "CrossJIT lookup MISS: kind=%d operation=%d types=%s,%s\n",
                key_kind, operation, left_type->tp_name, right_type->tp_name);
    }
    return provider;
}

int
_PyCrossJit_RecognizeBinary(JitOptContext *ctx, int operation,
                           JitOptRef left, JitOptRef right,
                           CrossJitRecognition *recognition)
{
    memset(recognition, 0, sizeof(*recognition));
    if (!_PyCrossJit_IsEnabled()) {
        return 0;
    }
    const PyJitProviderSpec *provider =
        lookup_type_pair(ctx, left, right, PYJIT_KEY_BINARY, operation);
    if (provider == NULL) {
        return 0;
    }
    JitOptRef operands[] = {left, right};
    int status = recognize_operation(ctx, provider, PYJIT_OPERATION_BINARY,
                                      operation, NULL, NULL, operands, 2, recognition);
    if (status > 0) {
        _PyInterpreterState_GET()->crossjit_stats.provider_first_recognized++;
    }
    return status;
}

void
_PyCrossJit_RecordSuppressedSpeculation(uint32_t count)
{
    _PyInterpreterState_GET()->crossjit_stats.scalar_speculations_avoided += count;
}

int
_PyCrossJit_BindBinary(JitOptContext *ctx, int operation,
                      JitOptRef left, JitOptRef right, JitOptRef result,
                      uint32_t uop_index,
                      const CrossJitRecognition *recognition)
{
    assert(recognition->provider != NULL);
    JitOptRef operands[] = {left, right};
    return bind_operation(ctx, operands, 2, result, uop_index, recognition);
}

int
_PyCrossJit_AnalyzeCompare(JitOptContext *ctx, int operation,
                           JitOptRef left, JitOptRef right, JitOptRef result,
                           uint32_t uop_index)
{
    if (!_PyCrossJit_IsEnabled()) {
        return 0;
    }
    const PyJitProviderSpec *provider =
        lookup_type_pair(ctx, left, right, PYJIT_KEY_COMPARE, operation);
    if (provider == NULL) {
        return 0;
    }
    JitOptRef operands[] = {left, right};
    return analyze_operation(ctx, provider, PYJIT_OPERATION_COMPARE, operation,
                             NULL, operands, 2, result, uop_index);
}

int
_PyCrossJit_AnalyzeUnary(JitOptContext *ctx, int operation,
                         JitOptRef operand, JitOptRef result,
                         uint32_t uop_index)
{
    if (!_PyCrossJit_IsEnabled()) {
        return 0;
    }
    PyTypeObject *type = crossjit_recognition_type(ctx, operand);
    const PyJitProviderSpec *provider = type == NULL ? NULL :
        _PyCrossJit_LookupUnaryProvider(_PyInterpreterState_GET(),
                                        type, operation);
    if (provider == NULL) {
        return 0;
    }
    JitOptRef operands[] = {operand};
    return analyze_operation(ctx, provider, PYJIT_OPERATION_UNARY, operation,
                             NULL, operands, 1, result, uop_index);
}

static int
crossjit_analyze_call(JitOptContext *ctx, PyObject *object,
                     JitOptRef *args, int arg_count, PyObject *keywords,
                     JitOptRef result, uint32_t uop_index)
{
    if (!_PyCrossJit_IsEnabled() || arg_count<1 || arg_count>(int)PYJIT_OPERATION_MAX_OPERANDS) return 0;
    const PyJitProviderSpec *provider=_PyCrossJit_LookupCallableProvider(_PyInterpreterState_GET(),object);
    if (!provider) return 0;
    if (keywords && (!PyTuple_CheckExact(keywords) || PyTuple_GET_SIZE(keywords)>arg_count)) return 0;
    CrossJitRecognition prepared={0};
    int status=recognize_operation(ctx,provider,PYJIT_OPERATION_CALL,0,object,keywords,args,(uint8_t)arg_count,&prepared);
    if (status<=0) return status;
    status=bind_operation(ctx,args,(uint8_t)arg_count,result,uop_index,&prepared);
    if (status<=0) return status;
    PyJitNodeRef *node=&ctx->crossjit->nodes[ctx->crossjit->node_count-1].handle;
    node->callable=object;
    node->keyword_names=keywords;
    node->positional_count=arg_count-(keywords?(uint8_t)PyTuple_GET_SIZE(keywords):0);
    return status;
}

int
_PyCrossJit_AnalyzeCall(JitOptContext *ctx, JitOptRef callable,
                        JitOptRef *args, int arg_count, JitOptRef result,
                        uint32_t uop_index)
{
    return crossjit_analyze_call(ctx,_Py_uop_sym_get_probable_value(callable),args,arg_count,NULL,result,uop_index);
}

int
_PyCrossJit_AnalyzeCallKw(JitOptContext *ctx,JitOptRef callable,JitOptRef *args,
                         int arg_count,JitOptRef names,JitOptRef result,uint32_t index)
{
    PyObject *keywords=_Py_uop_sym_get_probable_value(names);
    if (!keywords) return 0;
    return crossjit_analyze_call(ctx,_Py_uop_sym_get_probable_value(callable),args,arg_count,keywords,result,index);
}

int
_PyCrossJit_AnalyzeMethodCall(JitOptContext *ctx,JitOptRef callable,JitOptRef self,
    JitOptRef *args,int count,PyObject *keywords,JitOptRef result,uint32_t index)
{
    if (!_PyCrossJit_IsEnabled()) return 0;
    PyObject *descriptor=_Py_uop_sym_get_probable_value(callable);
    PyObject *receiver=_Py_uop_sym_get_probable_value(self);
    PyObject *bound=crossjit_registered_method(descriptor,receiver);
    if (!bound) return 0;
    int status=crossjit_analyze_call(ctx,bound,args,count,keywords,result,index);
    if (status>0) {
        PyJitNodeRef *node=&ctx->crossjit->nodes[ctx->crossjit->node_count-1].handle;
        node->physical_callable=descriptor;
        node->receiver=receiver;
        node->call_binding_kind=1;
    }
    return status;
}

/* Returns whether a node may be an interior virtual operation in a region. */
static int
node_is_virtualizable(const CrossJitNode *node)
{
    const PyJitSemanticSummary *summary = &node->handle.summary;
    return summary->mutation == PYJIT_MUTATION_NONE &&
           summary->result_alias == PYJIT_ALIAS_FRESH &&
           summary->materialization == PYJIT_MATERIALIZATION_VIRTUALIZABLE &&
           summary->replayability == PYJIT_REPLAYABLE &&
           !summary->may_call_host &&
           !summary->may_release_gil;
}

/* Returns whether a following node consumes a definition's result. */
static int
node_uses_result(const CrossJitNode *node, const CrossJitNode *definition)
{
    for (uint8_t i = 0; i < node->operand_count; i++) {
        if (definition->result_count == 1 &&
            node->operands[i] == definition->results[0]) {
            return 1;
        }
    }
    return 0;
}

/*
 * Proves that every non-final result in a candidate is both virtualizable and
 * consumed later in the same candidate.  This accepts general single-output
 * DAGs, including sibling computations that only meet at a later node, while
 * keeping region discovery independent of provider-specific semantics.
 */
static int
region_is_closed_dag(const CrossJitAnalysisState *state,
                     size_t first, size_t end)
{
    for (size_t definition = first; definition + 1 < end; definition++) {
        const CrossJitNode *node = &state->nodes[definition];
        if (!node_is_virtualizable(node)) {
            if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                fprintf(stderr, "CrossJIT formation boundary: nonvirtual node=%u semantic=%llu\n",
                        node->handle.id, (unsigned long long)node->handle.semantic_id);
            }
            return 0;
        }
        int has_internal_use = 0;
        for (size_t use = definition + 1; use < end; use++) {
            if (node_uses_result(&state->nodes[use], node)) {
                has_internal_use = 1;
                break;
            }
        }
        if (!has_internal_use) {
            if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                fprintf(stderr, "CrossJIT formation boundary: open definition node=%u semantic=%llu\n",
                        node->handle.id, (unsigned long long)node->handle.semantic_id);
            }
            return 0;
        }
    }
    return 1;
}

/* Returns a raw pre-register-allocation opcode for structural checks. */
static uint16_t
crossjit_base_opcode(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = instruction->opcode;
    if (opcode <= MAX_UOP_REGS_ID && _PyUop_Uncached[opcode] != 0) {
        opcode = _PyUop_Uncached[opcode];
    }
    return opcode;
}

/* Computes the number of values pushed by a uop from generated metadata. */
static int
crossjit_uop_pushed(uint16_t opcode, int oparg)
{
    /* Variable-arity vectorcalls consume their argument span and publish one
     * value. Their caching entry has no encoded oparg-dependent stack delta. */
    if (opcode==_CALL_NON_PY_GENERAL || opcode==_CALL_KW_NON_PY) return 1;
    if (opcode == _NOP || opcode == _CHECK_VALIDITY || opcode == _SET_IP ||
        opcode == _SPILL_OR_RELOAD ||
        opcode == _GUARD_TOS_FLOAT || opcode == _GUARD_NOS_FLOAT ||
        (opcode >= _RECORD_3OS_GEN_FUNC && opcode <= _RECORD_TOS_TYPE)) {
        return 0;
    }
    if (opcode == _POP_TOP || opcode == _POP_TOP_NOP ||
        opcode == _POP_TOP_INT || opcode == _POP_TOP_FLOAT ||
        opcode == _POP_TOP_UNICODE || opcode == _POP_TOP_OPARG) {
        return 0;
    }
    int popped = _PyUop_num_popped(opcode, oparg);
    if (popped < 0 || opcode > MAX_UOP_ID) {
        return -1;
    }
    const _PyUopCachingInfo *caching = &_PyUop_Caching[opcode];
    for (int input = 0; input <= MAX_CACHED_REGISTER; input++) {
        if (caching->entries[input].opcode != 0 &&
            caching->entries[input].output >= 0) {
            return popped + caching->entries[input].output - input;
        }
    }
    return -1;
}

/* Identifies loads that may be moved before the region's guard batch. */
static int
crossjit_is_safe_load(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    return opcode == _LOAD_FAST ||
           (opcode >= _LOAD_FAST_0 && opcode <= _LOAD_FAST_7) ||
           opcode == _LOAD_FAST_BORROW ||
           (opcode >= _LOAD_FAST_BORROW_0 && opcode <= _LOAD_FAST_BORROW_7) ||
           opcode == _LOAD_CONST_INLINE || opcode == _LOAD_CONST_INLINE_BORROW ||
           opcode == _PUSH_NULL || opcode == _LOAD_SMALL_INT ||
           (opcode >= _LOAD_SMALL_INT_0 && opcode <= _LOAD_SMALL_INT_3);
}

static PyObject *
crossjit_safe_load_constant(const _PyUOpInstruction *instruction)
{
    uint16_t opcode=crossjit_base_opcode(instruction);
    if (opcode==_PUSH_NULL) return NULL;
    if (opcode==_LOAD_SMALL_INT || (opcode>=_LOAD_SMALL_INT_0 && opcode<=_LOAD_SMALL_INT_3)) {
        unsigned int index=opcode==_LOAD_SMALL_INT?instruction->oparg:opcode-_LOAD_SMALL_INT_0;
        return index<_PY_NSMALLPOSINTS?(PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS+index]:NULL;
    }
    return (PyObject *)instruction->operand0;
}

/* Canonicalizes equivalent local/constant loads for shadow value identity. */
static uint64_t
crossjit_safe_load_key(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    if (opcode >= _LOAD_FAST_0 && opcode <= _LOAD_FAST_7) {
        return UINT64_C(0x100000000) + opcode - _LOAD_FAST_0;
    }
    if (opcode >= _LOAD_FAST_BORROW_0 && opcode <= _LOAD_FAST_BORROW_7) {
        return UINT64_C(0x100000000) + opcode - _LOAD_FAST_BORROW_0;
    }
    if (opcode == _LOAD_FAST || opcode == _LOAD_FAST_BORROW) {
        return UINT64_C(0x100000000) + instruction->oparg;
    }
    if (opcode==_PUSH_NULL) return UINT64_C(0x300000000);
    return UINT64_C(0x200000000) ^ (uintptr_t)crossjit_safe_load_constant(instruction);
}

/* Returns a frame-local index, or -1 for non-local safe loads. */
static int
crossjit_safe_load_local(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    if (opcode >= _LOAD_FAST_0 && opcode <= _LOAD_FAST_7) {
        return opcode - _LOAD_FAST_0;
    }
    if (opcode >= _LOAD_FAST_BORROW_0 && opcode <= _LOAD_FAST_BORROW_7) {
        return opcode - _LOAD_FAST_BORROW_0;
    }
    if (opcode == _LOAD_FAST || opcode == _LOAD_FAST_BORROW) {
        return instruction->oparg;
    }
    return opcode==_PUSH_NULL?-2:-1;
}

/* Generic local movement, not Provider domain semantics.  Admission here only
 * discovers a candidate; physical proof must independently establish an internal
 * virtual definition and a NULL local at both boundaries. */
static int
crossjit_local_store(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    if (opcode >= _SWAP_FAST_0 && opcode <= _SWAP_FAST_7) {
        return opcode - _SWAP_FAST_0;
    }
    return opcode == _SWAP_FAST ? instruction->oparg : -1;
}

static int
crossjit_is_local_movement(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    return crossjit_local_store(instruction) >= 0 ||
           opcode == _LOAD_FAST_AND_CLEAR || opcode == _DELETE_FAST;
}

/* Identifies the reference-cleanup uops emitted after generic BINARY_OP. */
static int
crossjit_is_cleanup(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    return opcode == _POP_TOP || opcode == _POP_TOP_NOP ||
           opcode == _POP_TOP_INT || opcode == _POP_TOP_FLOAT ||
           opcode == _POP_TOP_UNICODE;
}

static int
crossjit_is_store_cleanup(const _PyUOpInstruction *buffer, int index)
{
    return index > 0 && crossjit_is_cleanup(&buffer[index]) &&
           crossjit_local_store(&buffer[index - 1]) >= 0;
}

/* Recorder uops only annotate the trace and have no runtime side effects. */
static int
crossjit_is_record(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    return opcode >= _RECORD_3OS_GEN_FUNC && opcode <= _RECORD_TOS_TYPE;
}

/* Only exact built-in type tests with identity-preserving stack effects.
 * Overflow, representation, callable and value-dependent guards stay intact.
 * These are CPython structural semantics, independent of any Provider. */
static PyTypeObject *
crossjit_exact_type_guard(uint16_t opcode, int *depth)
{
    switch (opcode) {
        case _GUARD_TOS_FLOAT: *depth = 1; return &PyFloat_Type;
        case _GUARD_NOS_FLOAT: *depth = 2; return &PyFloat_Type;
        default: return NULL;
    }
}

/* These operations only copy/reorder stack references. Their value identity
 * is modeled below; no local write or observable operation is admitted. */
static int
crossjit_is_stack_movement(uint16_t opcode)
{
    return opcode == _COPY || opcode == _SWAP || opcode == _RROT_3;
}

/* Clears a rewritten instruction while retaining its buffer slot. */
static void
crossjit_make_nop(_PyUOpInstruction *instruction)
{
    memset(instruction, 0, sizeof(*instruction));
    instruction->opcode = _NOP;
}

/* Emits an opt-in structural rejection reason for implementation diagnostics. */
static int
crossjit_reject(const char *reason)
{
    _PyInterpreterState_GET()->crossjit_stats.rejected_regions++;
    if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
        fprintf(stderr, "CrossJIT region rejected: %s\n", reason);
    }
    return 0;
}

/* Looks up the Host-owned handle for one semantic value identity. */
static PyJitValueRef *
value_for_id(const CrossJitAnalysisState *state, uint32_t value_id)
{
    for (size_t i = 0; i < state->binding_count; i++) {
        if (state->bindings[i].value.id == value_id) {
            return (PyJitValueRef *)&state->bindings[i].value;
        }
    }
    return NULL;
}

/* Records each unique operand that is defined outside the candidate region. */
static int
region_collect_inputs(const CrossJitAnalysisState *state, CrossJitRegion *region)
{
    uint32_t defined[PYJIT_REGION_MAX_NODES];
    for (uint8_t i = 0; i < region->node_count; i++) {
        const CrossJitNode *node = &state->nodes[region->first_node + i];
        if (node->result_count != 1) {
            return 0;
        }
        defined[i] = node->results[0];
    }
    for (uint8_t i = 0; i < region->node_count; i++) {
        const CrossJitNode *node = &state->nodes[region->first_node + i];
        for (uint8_t operand = 0; operand < node->operand_count; operand++) {
            uint32_t value_id = node->operands[operand];
            bool is_internal = false;
            /* Only an earlier definition can satisfy an SSA operand. */
            for (uint8_t definition = 0; definition < i; definition++) {
                if (defined[definition] == value_id) {
                    is_internal = true;
                    break;
                }
            }
            if (is_internal) {
                continue;
            }
            bool already_input = false;
            for (uint8_t input = 0; input < region->input_count; input++) {
                if (region->inputs[input].value->id == value_id) {
                    already_input = true;
                    break;
                }
            }
            if (already_input) {
                continue;
            }
            if (region->input_count == PYJIT_REGION_MAX_LIVE_INS) {
                return 0;
            }
            PyJitValueRef *value = value_for_id(state, value_id);
            if (value == NULL) {
                return 0;
            }
            region->inputs[region->input_count++] = (PyJitRegionValue){value, 0, 0};
        }
    }
    return 1;
}

static uint64_t
fingerprint_mix(uint64_t hash, uint64_t value)
{
    hash ^= value;
    return hash * UINT64_C(1099511628211);
}

/* Hashes only Provider identity, semantic topology and normalized S_H.
 * Observations and every C_P/domain property are intentionally absent. */
static PyObject *
region_semantic_descriptor(const CrossJitAnalysisState *state,
                           const CrossJitRegion *region,
                           uint64_t *fingerprint)
{
    /* A compact normalized descriptor is the authoritative equality key.
     * The 64-bit fingerprint is diagnostic/Provider-facing only and never
     * defines semantic equivalence by itself. */
    uint64_t words[4 + PYJIT_REGION_MAX_NODES *
                         (13 + PYJIT_OPERATION_MAX_OPERANDS)];
    size_t used = 0;
    uint64_t hash = UINT64_C(1469598103934665603);
#define APPEND_DESCRIPTOR(value) do {                                     \
        uint64_t descriptor_value = (uint64_t)(value);                    \
        assert(used < Py_ARRAY_LENGTH(words));                            \
        words[used++] = descriptor_value;                                 \
        hash = fingerprint_mix(hash, descriptor_value);                   \
    } while (0)
    APPEND_DESCRIPTOR((uintptr_t)region->provider);
    APPEND_DESCRIPTOR(region->node_count);
    APPEND_DESCRIPTOR(region->input_count);
    APPEND_DESCRIPTOR(region->result_count);
    for (uint8_t i = 0; i < region->node_count; i++) {
        const CrossJitNode *node = &state->nodes[region->first_node + i];
        const PyJitSemanticSummary *s = &node->handle.summary;
        APPEND_DESCRIPTOR(node->handle.semantic_id);
        APPEND_DESCRIPTOR(node->operand_count);
        APPEND_DESCRIPTOR(node->result_count);
        APPEND_DESCRIPTOR(s->mutation);
        APPEND_DESCRIPTOR(s->result_alias);
        APPEND_DESCRIPTOR(s->materialization);
        APPEND_DESCRIPTOR(s->replayability);
        APPEND_DESCRIPTOR(s->may_raise);
        APPEND_DESCRIPTOR(s->may_call_host);
        APPEND_DESCRIPTOR(s->may_release_gil);
        APPEND_DESCRIPTOR((uintptr_t)s->exact_result_type);
        APPEND_DESCRIPTOR(node->handle.positional_count);
        /* Keyword tuples are immutable and pinned by the lowered plan.  An
         * identity mismatch deliberately prevents semantic-proof reuse. */
        APPEND_DESCRIPTOR((uintptr_t)node->handle.keyword_names);
        for (uint8_t operand = 0; operand < node->operand_count; operand++) {
            uint64_t topology = UINT64_MAX;
            for (uint8_t definition = 0; definition < i; definition++) {
                const CrossJitNode *def =
                    &state->nodes[region->first_node + definition];
                if (def->result_count == 1 &&
                    def->results[0] == node->operands[operand]) {
                    topology = definition;
                    break;
                }
            }
            if (topology == UINT64_MAX) {
                for (uint8_t input = 0; input < region->input_count; input++) {
                    if (region->inputs[input].value->id ==
                        node->operands[operand]) {
                        topology = UINT64_C(0x10000) + input;
                        break;
                    }
                }
            }
            APPEND_DESCRIPTOR(topology);
        }
    }
#undef APPEND_DESCRIPTOR
    *fingerprint = hash;
    return PyBytes_FromStringAndSize(
        (const char *)words, (Py_ssize_t)(used * sizeof(words[0])));
}

/* Records exact semantic-class reuse.  Semantic proof computation is still
 * performed for each trace; only this immutable classification is reused.
 * The cached descriptor is independent of C_P, so a new private domain
 * contract rebuilds Provider specialization without changing the class. */
static int
host_proof_cache_observe(CrossJitAnalysisState *state,
                         CrossJitRegion *region)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    interp->crossjit_stats.host_regions_considered++;
    PyObject *key = region_semantic_descriptor(
        state, region, &region->proof.structural_fingerprint);
    if (key == NULL) {
        return -1;
    }
    if (interp->crossjit_host_proof_cache == NULL) {
        interp->crossjit_host_proof_cache = PyDict_New();
        if (interp->crossjit_host_proof_cache == NULL) {
            Py_DECREF(key);
            return -1;
        }
    }
    if (PyDict_Size(interp->crossjit_host_proof_cache) > 4096) {
        PyDict_Clear(interp->crossjit_host_proof_cache);
    }
    int contains = PyDict_Contains(interp->crossjit_host_proof_cache, key);
    if (contains < 0) {
        Py_DECREF(key);
        return -1;
    }
    region->proof.semantic_class_reused = (uint8_t)contains;
    if (contains) {
        interp->crossjit_stats.semantic_class_reused++;
        interp->crossjit_stats.semantic_class_hit++;
    }
    else {
        if (PyDict_SetItem(interp->crossjit_host_proof_cache,
                           key, Py_None) < 0) {
            Py_DECREF(key);
            return -1;
        }
        interp->crossjit_stats.semantic_class_created++;
        interp->crossjit_stats.semantic_class_miss++;
    }
    Py_DECREF(key);
    interp->crossjit_stats.host_regions_proven++;
    return 0;
}

/*
 * Checks the provider-neutral trace contract between two semantic nodes.
 * Region formation stops at an observable operation instead of accepting a
 * larger DAG that the later structural rewrite would necessarily reject.
 */
static int
region_nodes_are_adjacent(const CrossJitNode *left,
                          const CrossJitNode *right,
                          const _PyUOpInstruction *buffer, int length)
{
    if (left->uop_index >= (uint32_t)length ||
        right->uop_index >= (uint32_t)length ||
        right->uop_index <= left->uop_index) {
        return 0;
    }
    uint16_t left_opcode = crossjit_base_opcode(&buffer[left->uop_index]);
    uint16_t right_opcode = crossjit_base_opcode(&buffer[right->uop_index]);
    int left_rewritable = left_opcode == _BINARY_OP ||
                          left_opcode == _UNARY_NEGATIVE;
    int right_rewritable = right_opcode == _BINARY_OP ||
                           right_opcode == _UNARY_NEGATIVE;
    if (!left_rewritable || !right_rewritable) {
        /* Calls have distinct stack contracts. Keep the semantic candidate;
         * the ABI4 physical proof independently validates its call mapping. */
        return 1;
    }
    int cleanup_count = 0;
    for (int i = (int)left->uop_index + 1; i < (int)right->uop_index; i++) {
        uint16_t opcode = crossjit_base_opcode(&buffer[i]);
        if (crossjit_is_cleanup(&buffer[i])) {
            cleanup_count += !crossjit_is_store_cleanup(buffer, i);
        }
        else if (opcode != _NOP && opcode != _CHECK_VALIDITY &&
                 opcode != _SET_IP && opcode != _SPILL_OR_RELOAD &&
                 !crossjit_is_stack_movement(opcode) &&
                 !crossjit_is_local_movement(&buffer[i]) &&
                 !crossjit_is_safe_load(&buffer[i]) &&
                 !crossjit_is_record(&buffer[i]) &&
                 crossjit_exact_type_guard(opcode, &(int){0}) == NULL) {
            if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                fprintf(stderr, "CrossJIT formation boundary: nodes=%u,%u separating_uop=%s\n",
                        left->handle.id, right->handle.id,
                        opcode <= MAX_UOP_ID ? _PyOpcode_uop_name[opcode] : "<invalid>");
            }
            return 0;
        }
    }
    return cleanup_count == left->operand_count;
}

/* Forms maximal same-provider, closed single-output DAGs before rewriting. */
int
_PyCrossJit_FormRegions(CrossJitAnalysisState *state,
                        _PyUOpInstruction *buffer, int length)
{
    if (state == NULL || buffer == NULL || length <= 0 ||
        state->node_count == 0) {
        return 0;
    }
    /* Evaluation-only scope ablation. Apply before closure, physical proof,
     * Provider admission and lowering, so every supported single operation
     * remains a candidate. Unset preserves the production formation policy. */
    const char *max_nodes_env = Py_GETENV("PYTHON_JIT_CROSSJIT_MAX_REGION_NODES");
    size_t max_nodes = max_nodes_env != NULL && strcmp(max_nodes_env, "1") == 0
        ? 1 : PYJIT_REGION_MAX_NODES;
    for (size_t first = 0; first < state->node_count; ) {
        CrossJitNode *head = &state->nodes[first];
        size_t minimum_nodes = PROVIDER_HAS(head->provider, supports_single_node_regions) &&
                               head->provider->supports_single_node_regions ? 1 : 2;
        size_t end = first + 1;
        while (end < state->node_count && end - first < max_nodes &&
               state->nodes[end].provider == head->provider &&
               region_nodes_are_adjacent(&state->nodes[end - 1],
                                         &state->nodes[end], buffer, length)) {
            end++;
        }
        size_t accepted_end = 0;
        while (end - first >= minimum_nodes) {
            if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                fprintf(stderr, "CrossJIT region candidate: first=%zu nodes=%zu\n",
                        first, end - first);
            }
            if (!region_is_closed_dag(state, first, end) ||
                (end - first == 1 && !node_is_virtualizable(head))) {
                end--;
                continue;
            }
            CrossJitRegion candidate = {
                .provider = head->provider,
                .first_node = (uint32_t)first,
                .node_count = (uint8_t)(end - first),
            };
            if (!region_collect_inputs(state, &candidate)) {
                end--;
                continue;
            }
            CrossJitNode *tail = &state->nodes[end - 1];
            candidate.result_count = 1;
            candidate.results[0] = (PyJitRegionValue){
                value_for_id(state, tail->results[0]), 1, 1
            };
            candidate.proof.node_count = candidate.node_count;
            candidate.proof.live_in_count = candidate.input_count;
            candidate.proof.result_count = 1;
            candidate.proof.virtual_value_count =
                (uint8_t)(candidate.node_count - 1);
            if (candidate.results[0].value == NULL ||
                host_proof_cache_observe(state, &candidate) < 0) {
                return -1;
            }
            if (grow((void **)&state->regions, sizeof(*state->regions),
                     &state->region_capacity, state->region_count + 1) < 0) {
                return -1;
            }
            state->regions[state->region_count++] = candidate;
            PyInterpreterState *interp = _PyInterpreterState_GET();
            interp->crossjit_stats.formed_regions++;
            interp->crossjit_stats.materialized_values++;
            interp->crossjit_stats.virtual_values += candidate.node_count - 1;
            accepted_end = end;
            break;
        }
        /* A rejected leading node must not hide a valid suffix region. */
        first = accepted_end != 0 ? accepted_end : first + 1;
    }
    return (int)state->region_count;
}

/* Destroys Provider plans and owned Host constants before publication. */
static void
crossjit_destroy_unpublished_plan(CrossJitExecutorPlan *plan)
{
    if (plan->provider != NULL && plan->guard_plan != NULL &&
        PROVIDER_HAS(plan->provider, destroy_guard_plan) &&
        plan->provider->destroy_guard_plan != NULL) {
        plan->provider->destroy_guard_plan(plan->guard_plan);
    }
    if (plan->provider != NULL && plan->exec_plan != NULL &&
        PROVIDER_HAS(plan->provider, destroy_exec_plan) &&
        plan->provider->destroy_exec_plan != NULL) {
        plan->provider->destroy_exec_plan(plan->exec_plan);
    }
    for (uint8_t input = 0; input < plan->input_count; input++) {
        Py_CLEAR(plan->input_constants[input]);
    }
    for (uint8_t j=0;j<plan->call_guard_count;j++) { Py_CLEAR(plan->call_guard_objects[j]); }
    plan->call_guard_count=0;
    memset(plan, 0, sizeof(*plan));
}

/* Returns whether a semantic value is a live-in available at the region
 * entry, which is the only placement implemented by the first v3 scheduler. */
static int
region_has_live_in(const PyJitRegionView *view,
                   const PyJitValueRef *dependency)
{
    for (uint8_t i = 0; i < view->region->input_count; i++) {
        if (view->region->inputs[i].value == dependency) {
            return 1;
        }
    }
    return 0;
}

static int
assumption_equal(const PyJitAssumption *left,
                 const PyJitAssumption *right)
{
    if (left->kind != right->kind ||
        left->dependency_count != right->dependency_count ||
        left->canonical_key_size != right->canonical_key_size ||
        (left->canonical_key_size != 0 &&
         memcmp(left->canonical_key, right->canonical_key,
                left->canonical_key_size) != 0)) {
        return 0;
    }
    for (uint8_t i = 0; i < left->dependency_count; i++) {
        if (left->dependencies[i]->id != right->dependencies[i]->id) {
            return 0;
        }
    }
    return 1;
}

/* Host-owned scheduling validates dependencies and deduplicates assumptions.
 * Providers supply guard meaning and implementation, never placement. */
static int
schedule_assumptions(const PyJitRegionView *view,
                     const PyJitPrepareResult *prepared,
                     PyJitAssumption **scheduled_out,
                     size_t *count_out)
{
    *scheduled_out = NULL;
    *count_out = 0;
    if (prepared->assumption_count == 0) {
        return 0;
    }
    if (prepared->assumptions == NULL ||
        prepared->assumption_count > PYJIT_REGION_MAX_LIVE_INS) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid CrossJIT assumption vector");
        return -1;
    }
    PyJitAssumption *scheduled = PyMem_Calloc(
        prepared->assumption_count, sizeof(*scheduled));
    if (scheduled == NULL) {
        return PyErr_NoMemory(), -1;
    }
    PyInterpreterState *interp = _PyInterpreterState_GET();
    for (size_t i = 0; i < prepared->assumption_count; i++) {
        const PyJitAssumption *assumption = &prepared->assumptions[i];
        interp->crossjit_stats.assumptions_before_dedup++;
        if (assumption->dependency_count >
                PYJIT_REGION_MAX_LIVE_INS ||
            (assumption->semantic_property_mask &
             ~PYJIT_SEMANTIC_PROPERTY_MASK) != 0 ||
            (assumption->dependency_count != 0 &&
             assumption->dependencies == NULL) ||
            (assumption->canonical_key_size != 0 &&
             assumption->canonical_key == NULL) ||
            (assumption->payload_size != 0 &&
             assumption->payload == NULL)) {
            PyMem_Free(scheduled);
            PyErr_SetString(PyExc_ValueError,
                            "invalid CrossJIT assumption");
            return -1;
        }
        for (uint8_t dependency = 0;
             dependency < assumption->dependency_count; dependency++) {
            if (assumption->dependencies[dependency] == NULL ||
                !region_has_live_in(
                    view, assumption->dependencies[dependency])) {
                PyMem_Free(scheduled);
                PyErr_SetString(PyExc_ValueError,
                                "CrossJIT guard dependency is unavailable at Host placement");
                return -1;
            }
        }
        int duplicate = 0;
        for (size_t existing = 0; existing < *count_out; existing++) {
            if (assumption_equal(&scheduled[existing], assumption)) {
                scheduled[existing].semantic_property_mask |=
                    assumption->semantic_property_mask;
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            scheduled[(*count_out)++] = *assumption;
            interp->crossjit_stats.assumptions_after_dedup++;
        }
    }
    *scheduled_out = scheduled;
    return 0;
}

/* Specializes only after Pi_H exists, then combines Pi_H with the opaque C_P
 * to create provider guard/execution plans. */
static int
crossjit_lower_region(const PyJitRegionView *view,
                      CrossJitExecutorPlan *plan,
                      PyJitPrepareDecision *decision_out)
{
    const PyJitProviderSpec *provider = view->region->provider;
    PyInterpreterState *interp = _PyInterpreterState_GET();
    memset(plan, 0, sizeof(*plan));
    *decision_out = PYJIT_PREPARE_ERROR;
    plan->provider = provider;
    plan->input_count = view->region->input_count;

    PyJitPrepareResult prepared = {
        .struct_size = sizeof(PyJitPrepareResult),
    };
    PyJitResult status = provider->prepare_region(view, &prepared);
    if (status == PYJIT_RESULT_ERROR ||
        prepared.decision == PYJIT_PREPARE_ERROR) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_SystemError,
                            "CrossJIT prepare_region failed without an exception");
        }
        return -1;
    }
    if (status != PYJIT_RESULT_HIT) {
        return 0;
    }
    *decision_out = prepared.decision;
    switch (prepared.decision) {
        case PYJIT_PREPARE_ACCEPT:
            break;
        case PYJIT_PREPARE_UNSUPPORTED_NODE:
            interp->crossjit_stats.reject_unsupported_node++;
            if (prepared.contract != NULL && provider->destroy_contract != NULL) {
                provider->destroy_contract(prepared.contract);
            }
            return 0;
        case PYJIT_PREPARE_UNSUPPORTED_CONTRACT:
            interp->crossjit_stats.reject_unsupported_contract++;
            if (prepared.contract != NULL && provider->destroy_contract != NULL) {
                provider->destroy_contract(prepared.contract);
            }
            return 0;
        case PYJIT_PREPARE_UNPROFITABLE:
            interp->crossjit_stats.reject_unprofitable++;
            if (prepared.contract != NULL && provider->destroy_contract != NULL) {
                provider->destroy_contract(prepared.contract);
            }
            return 0;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "invalid CrossJIT prepare decision");
            return -1;
    }
    if (prepared.contract == NULL && prepared.contract_size != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "CrossJIT contract has size but no storage");
        return -1;
    }
    if (view->region->proof.semantic_class_reused) {
        interp->crossjit_stats.provider_specialization_rebuilt++;
    }
    else {
        interp->crossjit_stats.provider_specialization_created++;
    }

    PyJitAssumption *scheduled = NULL;
    size_t scheduled_count = 0;
    int schedule_status = schedule_assumptions(
        view, &prepared, &scheduled, &scheduled_count);
    if (schedule_status < 0) {
        if (prepared.contract != NULL && provider->destroy_contract != NULL) {
            provider->destroy_contract(prepared.contract);
        }
        return -1;
    }
    PyJitSpecializationView specialization = {
        .region = view,
        .contract = prepared.contract,
        .contract_size = prepared.contract_size,
        .assumptions = scheduled,
        .assumption_count = scheduled_count,
    };

    int lowered = 0;
    if (scheduled_count != 0) {
        if (provider->lower_guards == NULL) {
            goto done;
        }
        status = provider->lower_guards(
            &specialization, &plan->guard_plan);
        if (status == PYJIT_RESULT_ERROR) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_SystemError,
                                "CrossJIT lower_guards failed without an exception");
            }
            lowered = -1;
            goto done;
        }
        if (status != PYJIT_RESULT_HIT || plan->guard_plan == NULL) {
            goto done;
        }
    }

    status = provider->lower_region(&specialization, &plan->exec_plan);
    if (status == PYJIT_RESULT_ERROR) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_SystemError,
                            "CrossJIT lower_region failed without an exception");
        }
        lowered = -1;
        goto done;
    }
    if (status == PYJIT_RESULT_HIT && plan->exec_plan != NULL) {
        plan->combined_guards = plan->guard_plan != NULL &&
            PROVIDER_HAS(provider, run_guarded_region) &&
            provider->run_guarded_region != NULL;
        lowered = 1;
    }

done:
    PyMem_Free(scheduled);
    if (prepared.contract != NULL && provider->destroy_contract != NULL) {
        provider->destroy_contract(prepared.contract);
    }
    if (lowered <= 0) {
        crossjit_destroy_unpublished_plan(plan);
    }
    return lowered;
}

static int
crossjit_is_call_guard(uint16_t opcode)
{
    return opcode==_GUARD_CALLABLE_BUILTIN_FAST_WITH_KEYWORDS ||
        opcode==_CHECK_IS_NOT_PY_CALLABLE_KW ||
        opcode==_GUARD_CALLABLE_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS;
}

/* Rewrites one proven unary/binary DAG without changing trace buffer length. */
static int
crossjit_rewrite_region(CrossJitAnalysisState *state, CrossJitRegion *region,
                        _PyUOpInstruction *buffer, int length)
{
    /* Pi_H^semantic may be a cache hit, but Pi_H^physical is deliberately
     * reconstructed from this trace on every rewrite attempt. */
    _PyInterpreterState_GET()->crossjit_stats.physical_proof_attempted++;
    state->last_lowering_decline=0;
    CrossJitNode *nodes[PYJIT_REGION_MAX_NODES];
    int exec_indices[PYJIT_REGION_MAX_NODES];
    int cleanup_pops[PYJIT_REGION_MAX_NODES];
    int first_call=0, first_keywords=0;
    int calls[PYJIT_REGION_MAX_NODES]={0}, keywords[PYJIT_REGION_MAX_NODES]={0}, fastcalls[PYJIT_REGION_MAX_NODES]={0};
    for (uint8_t i = 0; i < region->node_count; i++) {
        nodes[i] = &state->nodes[region->first_node + i];
        if (nodes[i]->uop_index >= (uint32_t)length) {
            return crossjit_reject("semantic uop index is outside the trace");
        }
        exec_indices[i] = (int)nodes[i]->uop_index;
        uint16_t opcode = crossjit_base_opcode(&buffer[exec_indices[i]]);
        int call=nodes[i]->handle.callable &&
            (opcode==_CALL_NON_PY_GENERAL || opcode==_CALL_KW_NON_PY ||
             opcode==_CALL_BUILTIN_FAST_WITH_KEYWORDS || opcode==_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS);
        calls[i]=call;
        keywords[i]=call && opcode==_CALL_KW_NON_PY;
        fastcalls[i]=call && (opcode==_CALL_BUILTIN_FAST_WITH_KEYWORDS || opcode==_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS);
        cleanup_pops[i]=call?((opcode==_CALL_BUILTIN_FAST_WITH_KEYWORDS || opcode==_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS)?nodes[i]->operand_count+1:0):nodes[i]->operand_count;
        if (i==0) {first_call=call;first_keywords=opcode==_CALL_KW_NON_PY;}
        if (!((opcode == _BINARY_OP && nodes[i]->operand_count == 2) ||
              (opcode == _UNARY_NEGATIVE && nodes[i]->operand_count == 1) || call) ||
            (i != 0 && exec_indices[i] <= exec_indices[i - 1])) {
            return crossjit_reject("region nodes are not ordered generic arithmetic uops");
        }
    }

    for (uint8_t node = 1; node < region->node_count; node++) {
        int cleanup_count = 0;
        for (int i = exec_indices[node - 1] + 1; i < exec_indices[node]; i++) {
            uint16_t opcode = crossjit_base_opcode(&buffer[i]);
            if (crossjit_base_opcode(&buffer[i])==_POP_TOP_OPARG) {
                cleanup_count += buffer[i].oparg;
            }
            else if (crossjit_is_cleanup(&buffer[i])) {
                cleanup_count += !crossjit_is_store_cleanup(buffer, i);
            }
            else if (opcode != _NOP && opcode != _CHECK_VALIDITY &&
                     opcode != _SET_IP && opcode != _SPILL_OR_RELOAD &&
                     !crossjit_is_stack_movement(opcode) &&
                     !crossjit_is_local_movement(&buffer[i]) &&
                     !crossjit_is_safe_load(&buffer[i]) &&
                     !crossjit_is_record(&buffer[i]) &&
                     !crossjit_is_call_guard(opcode) &&
                     opcode!=_TIER2_RESUME_CHECK &&
                     crossjit_exact_type_guard(opcode, &(int){0}) == NULL) {
                if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                    fprintf(stderr, "CrossJIT separating uop %d: %s\n", i,
                            opcode <= MAX_UOP_ID ? _PyOpcode_uop_name[opcode] :
                            "<invalid>");
                }
                return crossjit_reject("effectful uop separates semantic nodes");
            }
        }
        /* Extra cleanup can belong to an intervening owned constant load.
         * The shadow stack below proves every pop and the final live values. */
        if (cleanup_count < cleanup_pops[node - 1]) {
            return crossjit_reject("arithmetic cleanup shape differs from the trace contract");
        }
    }
    int last_exec = exec_indices[region->node_count - 1];
    int remaining_cleanup=cleanup_pops[region->node_count-1];
    int last=last_exec;
    while (remaining_cleanup>0) {
        if (++last>=length) return crossjit_reject("final cleanup outside trace");
        uint16_t op=crossjit_base_opcode(&buffer[last]);
        if (op==_POP_TOP_OPARG) remaining_cleanup-=buffer[last].oparg;
        else if (crossjit_is_cleanup(&buffer[last])) remaining_cleanup--;
        else return crossjit_reject("final call/arithmetic cleanup incomplete");
    }
    if (remaining_cleanup) return crossjit_reject("final cleanup exceeded operands");
    int first_exec = exec_indices[0];
    int first_set_ip = first_exec - 1;
    while (first_set_ip >= 0 &&
           (crossjit_base_opcode(&buffer[first_set_ip]) == _NOP ||
            crossjit_is_record(&buffer[first_set_ip]) ||
            crossjit_base_opcode(&buffer[first_set_ip]) == _SPILL_OR_RELOAD ||
            (first_call && crossjit_is_call_guard(crossjit_base_opcode(&buffer[first_set_ip]))) ||
            crossjit_exact_type_guard(crossjit_base_opcode(&buffer[first_set_ip]),
                                     &(int){0}) != NULL)) {
        first_set_ip--;
    }
    if (first_set_ip < 0 ||
        crossjit_base_opcode(&buffer[first_set_ip]) != _SET_IP) {
        return crossjit_reject("region has no dominating instruction pointer");
    }

    int stack[UOP_MAX_TRACE_LENGTH * 4];
    int stack_at_first[UOP_MAX_TRACE_LENGTH * 4];
    int depth = 0;
    int depth_at_first = -1;
    int next_value = 1;
    int node_inputs[PYJIT_REGION_MAX_NODES][PYJIT_OPERATION_MAX_OPERANDS] = {{0}};
    int node_outputs[PYJIT_REGION_MAX_NODES] = {0};
    int call_values[PYJIT_REGION_MAX_NODES][3]={{0}};
    int load_count = 0;
    uint64_t load_keys[PYJIT_REGION_MAX_LIVE_INS * 2] = {0};
    int load_runtime_values[PYJIT_REGION_MAX_LIVE_INS * 2] = {0};
    int load_locals[PYJIT_REGION_MAX_LIVE_INS * 2] = {0};
    PyObject *load_constants[PYJIT_REGION_MAX_LIVE_INS * 2] = {0};
    int known_load_count = 0;
    int16_t transient_locals[PYJIT_REGION_MAX_NODES];
    int transient_values[PYJIT_REGION_MAX_NODES] = {0};
    int transient_count = 0;
    int guard_values[PYJIT_REGION_MAX_NODES * 2];
    PyTypeObject *guard_types[PYJIT_REGION_MAX_NODES * 2];
    int guard_count = 0;
    uint8_t current_node = 0;
    for (int i = 0; i <= last; i++) {
        uint16_t opcode = crossjit_base_opcode(&buffer[i]);
        if (i < first_exec && opcode != _NOP && opcode != _CHECK_VALIDITY &&
            opcode != _SET_IP && opcode != _SPILL_OR_RELOAD &&
            !crossjit_is_safe_load(&buffer[i]) &&
            !crossjit_is_cleanup(&buffer[i]) &&
            !crossjit_is_record(&buffer[i]) &&
            !crossjit_is_call_guard(opcode) && opcode!=_TIER2_RESUME_CHECK &&
            opcode!=_PUSH_NULL &&
            !crossjit_is_stack_movement(opcode) &&
            crossjit_exact_type_guard(opcode, &(int){0}) == NULL) {
            /* Earlier local/frame changes and calls cannot share a local-load
             * epoch with the region. Stack values themselves retain identity. */
            /* Owned inline constants keep their identity across prefix
             * epochs. Only local loads can be invalidated by prior calls. */
            int retained=0;
            for (int known=0;known<known_load_count;known++) {
                if (load_constants[known]==NULL && load_locals[known]!=-2) continue;
                load_keys[retained]=load_keys[known];
                load_runtime_values[retained]=load_runtime_values[known];
                load_locals[retained]=load_locals[known];
                load_constants[retained++]=load_constants[known];
            }
            known_load_count = retained;
        }
        if (i>=first_set_ip && (crossjit_is_call_guard(opcode) || opcode==_TIER2_RESUME_CHECK)) continue;
        int popped = _PyUop_num_popped(opcode, buffer[i].oparg);
        int pushed = crossjit_uop_pushed(opcode, buffer[i].oparg);
        int guard_depth = 0;
        PyTypeObject *guard_type = crossjit_exact_type_guard(opcode, &guard_depth);
        int prefix_needed = guard_type == NULL ? popped : guard_depth;
        if (i < first_exec && popped >= 0 && depth < prefix_needed) {
            /* A trace can start with live stack references (for example a
             * loop-exhaustion side trace starts with iterator/index). Their
             * entry depth is not part of this region's input contract.
             * Introduce opaque identities below the known prefix stack;
             * the region proof still requires every consumed value to map
             * to its actual stack/local source and forbids underflow inside
             * the region. Unused deeper entry references remain untouched. */
            int missing = prefix_needed - depth;
            if (depth + missing >= (int)Py_ARRAY_LENGTH(stack)) {
                return crossjit_reject("prefix entry stack exceeds simulation capacity");
            }
            memmove(stack + missing, stack, (size_t)depth * sizeof(stack[0]));
            for (int entry = 0; entry < missing; entry++) {
                stack[entry] = next_value++;
            }
            depth += missing;
        }
        if (popped < 0 || pushed < 0 || depth < popped ||
            depth - popped + pushed >= (int)Py_ARRAY_LENGTH(stack)) {
            if (Py_GETENV("PYTHON_JIT_CROSSJIT_DEBUG") != NULL) {
                fprintf(stderr,
                        "CrossJIT stack simulation failed at %d: %s "
                        "depth=%d popped=%d pushed=%d\n", i,
                        opcode <= MAX_UOP_ID ? _PyOpcode_uop_name[opcode] :
                        "<invalid>", depth, popped, pushed);
            }
            return crossjit_reject("uop stack effect cannot be simulated");
        }
        if (guard_type != NULL) {
            /* Metadata counts zero consumed/produced values for an in-place
             * observation of existing stack references. */
            if (depth < guard_depth || popped != 0 || pushed != 0) {
                return crossjit_reject("type guard stack effect is not identity-preserving");
            }
            if (i >= first_set_ip) {
                int value = stack[depth - guard_depth];
                int duplicate = 0;
                for (int guard = 0; guard < guard_count; guard++) {
                    duplicate |= guard_values[guard] == value &&
                                 guard_types[guard] == guard_type;
                }
                if (!duplicate) {
                    if (guard_count == (int)Py_ARRAY_LENGTH(guard_values)) {
                        return crossjit_reject("too many structural type guards");
                    }
                    guard_values[guard_count] = value;
                    guard_types[guard_count++] = guard_type;
                }
            }
            /* The original guard observes, but never replaces, the value. */
            continue;
        }
        if (opcode == _COPY) {
            if (buffer[i].oparg < 1 || depth < buffer[i].oparg) {
                return crossjit_reject("invalid stack copy depth");
            }
            stack[depth] = stack[depth - buffer[i].oparg];
            depth++;
            continue;
        }
        if (opcode == _SWAP) {
            if (buffer[i].oparg < 2 || depth < buffer[i].oparg) {
                return crossjit_reject("invalid stack swap depth");
            }
            int temporary = stack[depth - 1];
            stack[depth - 1] = stack[depth - buffer[i].oparg];
            stack[depth - buffer[i].oparg] = temporary;
            continue;
        }
        if (opcode == _RROT_3) {
            if (depth < 3) {
                return crossjit_reject("invalid stack rotation depth");
            }
            int temporary = stack[depth - 1];
            stack[depth - 1] = stack[depth - 2];
            stack[depth - 2] = stack[depth - 3];
            stack[depth - 3] = temporary;
            continue;
        }
        if (i > first_exec) {
            int local = crossjit_local_store(&buffer[i]);
            int is_store = local >= 0;
            if (!is_store && (opcode == _DELETE_FAST ||
                              opcode == _LOAD_FAST_AND_CLEAR)) {
                local = buffer[i].oparg;
            }
            if (local >= 0) {
                int slot = 0;
                while (slot < transient_count && transient_locals[slot] != local) {
                    slot++;
                }
                if (slot == transient_count) {
                    if (!is_store || local >= INT16_MAX ||
                        transient_count == PYJIT_REGION_MAX_NODES) {
                        return crossjit_reject("local movement has no virtual definition");
                    }
                    /* An entry local cannot simultaneously be a DAG live-in.
                     * Otherwise erasing its store would change later loads. */
                    for (int known = 0; known < known_load_count; known++) {
                        if (load_locals[known] == local) {
                            return crossjit_reject("temporary local aliases a live-in load");
                        }
                    }
                    transient_locals[transient_count++] = (int16_t)local;
                }
                if (is_store) {
                    int definition = 0;
                    for (uint8_t node = 0; node < current_node; node++) {
                        definition |= node_outputs[node] == stack[depth - 1];
                    }
                    if (!definition || i + 1 > last ||
                        !crossjit_is_store_cleanup(buffer, i + 1)) {
                        return crossjit_reject("local store is not a virtual temporary");
                    }
                    transient_values[slot] = stack[depth - 1];
                    /* _SWAP_FAST leaves its old (guarded NULL) reference for
                     * cleanup. A replaced virtual temporary is also private. */
                    stack[depth - 1] = next_value++;
                }
                else {
                    if (transient_values[slot] == 0) {
                        return crossjit_reject("cleared temporary local read twice");
                    }
                    if (opcode == _LOAD_FAST_AND_CLEAR) {
                        stack[depth++] = transient_values[slot];
                    }
                    transient_values[slot] = 0;
                }
                continue;
            }
            local = crossjit_safe_load_local(&buffer[i]);
            for (int slot = 0; slot < transient_count; slot++) {
                if (local == transient_locals[slot]) {
                    if (transient_values[slot] == 0) {
                        return crossjit_reject("load of cleared virtual local");
                    }
                    stack[depth++] = transient_values[slot];
                    goto simulated_local_load;
                }
            }
        }
        int is_node = current_node < region->node_count &&
                      i == exec_indices[current_node];
        if (i == first_exec) {
            depth_at_first = depth;
            memcpy(stack_at_first, stack, (size_t)depth * sizeof(int));
        }
        if (is_node) {
            /* An exception is an observer of frame locals. Until a Provider
             * can supply reconstruction for an ERROR, no fallible operation
             * may execute with a suppressed local binding still live. */
            if (nodes[current_node]->handle.summary.may_raise) {
                for (int slot = 0; slot < transient_count; slot++) {
                    if (transient_values[slot] != 0) {
                        return crossjit_reject("fallible operation observes a virtual local");
                    }
                }
            }
            uint8_t operand_count = nodes[current_node]->operand_count;
            int kw=keywords[current_node];
            int call=calls[current_node];
            int fast=fastcalls[current_node];
            int expected_pops=call?operand_count+2+kw:operand_count;
            if (depth<expected_pops || (fast?(popped!=0 || pushed!=0):(popped!=expected_pops || pushed<1))) {
                return crossjit_reject("semantic operation stack effect unexpected");
            }
            for (uint8_t operand=0;operand<operand_count;operand++) {
                node_inputs[current_node][operand]=stack[depth-operand_count-kw+operand];
            }
            if (call) {
                call_values[current_node][0]=stack[depth-expected_pops];
                call_values[current_node][1]=stack[depth-expected_pops+1];
                if (kw) call_values[current_node][2]=stack[depth-1];
            }
            if (fast) {
                int value=next_value++;
                stack[depth-operand_count-2]=value;
                node_outputs[current_node]=value;
                current_node++;
                continue;
            }
        }
        depth -= popped;
        for (int output = 0; output < pushed; output++) {
            int value = next_value++;
            int repeated_load = 0;
            if (output == 0 && crossjit_is_safe_load(&buffer[i])) {
                uint64_t key = crossjit_safe_load_key(&buffer[i]);
                for (int known = 0; known < known_load_count; known++) {
                    if (load_keys[known] == key) {
                        value = load_runtime_values[known];
                        repeated_load = 1;
                        break;
                    }
                }
                if (!repeated_load &&
                    known_load_count < (int)Py_ARRAY_LENGTH(load_keys)) {
                    load_keys[known_load_count] = key;
                    load_runtime_values[known_load_count++] = value;
                    load_locals[known_load_count - 1] =
                        crossjit_safe_load_local(&buffer[i]);
                    load_constants[known_load_count - 1] =
                        load_locals[known_load_count - 1] < 0 ?
                        crossjit_safe_load_constant(&buffer[i]) : NULL;
                }
            }
            if (is_node && output == 0) {
                node_outputs[current_node] = value;
            }
            if (i > first_exec && i < last_exec && output == 0 &&
                crossjit_is_safe_load(&buffer[i]) && !repeated_load) {
                if (load_count == PYJIT_REGION_MAX_LIVE_INS) {
                    return crossjit_reject("too many movable live-in loads");
                }
                load_count++;
            }
            stack[depth++] = value;
        }
        if (is_node) {
            current_node++;
        }
simulated_local_load:;
    }
    if (current_node != region->node_count || depth_at_first < 0) {
        return crossjit_reject("not all semantic nodes were found in the trace");
    }
    for (int slot = 0; slot < transient_count; slot++) {
        if (transient_values[slot] != 0) {
            return crossjit_reject("virtual temporary local remains live at region exit");
        }
    }
    uint32_t semantic_ids[PYJIT_REGION_MAX_LIVE_INS + PYJIT_REGION_MAX_NODES];
    int runtime_ids[PYJIT_REGION_MAX_LIVE_INS + PYJIT_REGION_MAX_NODES];
    int mapping_count = 0;
    for (uint8_t node = 0; node < region->node_count; node++) {
        for (uint8_t operand = 0; operand < nodes[node]->operand_count; operand++) {
            uint32_t semantic = nodes[node]->operands[operand];
            int existing = -1;
            for (int i = 0; i < mapping_count; i++) {
                if (semantic_ids[i] == semantic) {
                    existing = i;
                    break;
                }
            }
            if (existing >= 0 && runtime_ids[existing] != node_inputs[node][operand]) {
                return crossjit_reject("semantic identity maps to multiple runtime values");
            }
            if (existing < 0) {
                semantic_ids[mapping_count] = semantic;
                runtime_ids[mapping_count++] = node_inputs[node][operand];
            }
        }
        semantic_ids[mapping_count] = nodes[node]->results[0];
        runtime_ids[mapping_count++] = node_outputs[node];
    }

    int expected_inputs[PYJIT_REGION_MAX_LIVE_INS];
    for (uint8_t input = 0; input < region->input_count; input++) {
        uint32_t semantic = region->inputs[input].value->id;
        expected_inputs[input] = -1;
        for (int i = 0; i < mapping_count; i++) {
            if (semantic_ids[i] == semantic) {
                expected_inputs[input] = runtime_ids[i];
                break;
            }
        }
        if (expected_inputs[input] < 0) {
            return crossjit_reject("semantic live-in has no runtime value");
        }
    }
    /* Include an already-materialized DAG input below the first operation's
     * operands when it is consumed later in the same region.  The exit check
     * below proves that every value in this physical span is dead afterwards. */
    int first_needed = depth_at_first;
    for (int slot = 0; slot < depth_at_first; slot++) {
        for (uint8_t input = 0; input < region->input_count; input++) {
            if (stack_at_first[slot] == expected_inputs[input] &&
                slot < first_needed) {
                first_needed = slot;
            }
        }
    }
    for (uint8_t node=0;node<region->node_count;node++) {
        if (!calls[node]) continue;
        for (int role=0;role<2+keywords[node];role++) {
            for (int slot=0;slot<depth_at_first;slot++) {
                if (stack_at_first[slot]==call_values[node][role] && slot<first_needed) first_needed=slot;
            }
        }
    }
    if (first_needed == depth_at_first) {
        return crossjit_reject("no semantic live-in is present at region entry");
    }
    int physical_count = depth_at_first - first_needed;
    if ((unsigned int)physical_count > PYJIT_REGION_MAX_LIVE_INS) {
        return crossjit_reject("too many physical entry values");
    }
    int physical_inputs[PYJIT_REGION_MAX_LIVE_INS];
    for (int slot = first_needed; slot < depth_at_first; slot++) {
        physical_inputs[slot - first_needed] = stack_at_first[slot];
    }
    int first_operand = physical_count - nodes[0]->operand_count - first_keywords;
    for (int input = 0; input < nodes[0]->operand_count; input++) {
        if (physical_inputs[first_operand + input] != node_inputs[0][input]) {
            return crossjit_reject("entry stack has unexpected arithmetic operands");
        }
    }
    int16_t input_sources[PYJIT_REGION_MAX_LIVE_INS];
    PyObject *input_constants[PYJIT_REGION_MAX_LIVE_INS] = {0};
    for (uint8_t input = 0; input < region->input_count; input++) {
        input_sources[input] = INT16_MIN;
        for (int physical = 0; physical < physical_count; physical++) {
            if (physical_inputs[physical] == expected_inputs[input]) {
                input_sources[input] = (int16_t)physical;
                break;
            }
        }
        if (input_sources[input] != INT16_MIN) {
            continue;
        }
        for (int known = 0; known < known_load_count; known++) {
            if (load_runtime_values[known] == expected_inputs[input]) {
                if (load_locals[known] >= 0 && load_locals[known] < INT16_MAX) {
                    input_sources[input] = (int16_t)(-load_locals[known] - 1);
                    break;
                }
                if (load_constants[known] != NULL) {
                    input_constants[input] = load_constants[known];
                    break;
                }
            }
        }
        if (input_sources[input] == INT16_MIN && input_constants[input] == NULL) {
            return crossjit_reject("semantic live-in is not on stack, in a local or constant");
        }
    }
    for (uint8_t node = 0; node + 1 < region->node_count; node++) {
        for (int slot = 0; slot < depth; slot++) {
            if (stack[slot] == node_outputs[node]) {
                return crossjit_reject("virtual intermediate remains live at region exit");
            }
        }
    }
    for (int physical = 0; physical < physical_count; physical++) {
        for (int slot = 0; slot < depth; slot++) {
            if (stack[slot] == physical_inputs[physical]) {
                return crossjit_reject(
                    "physical entry value remains live at region exit");
            }
        }
    }
    int final_uses = 0;
    for (int slot = 0; slot < depth; slot++) {
        final_uses += stack[slot] == node_outputs[region->node_count - 1];
    }
    if (final_uses != 1) {
        return crossjit_reject("region final result is not a single live value");
    }
    uint8_t guard_inputs[PYJIT_REGION_MAX_NODES * 2];
    for (int guard = 0; guard < guard_count; guard++) {
        int input = 0;
        while (input < region->input_count &&
               expected_inputs[input] != guard_values[guard]) {
            input++;
        }
        if (input == region->input_count) {
            return crossjit_reject("structural type guard depends on a non-live-in value");
        }
        guard_inputs[guard] = (uint8_t)input;
    }

    int16_t call_sources[PYJIT_REGION_MAX_NODES*3];
    PyObject *call_objects[PYJIT_REGION_MAX_NODES*3];
    int call_count=0;
    for (uint8_t node=0;node<region->node_count;node++) {
        if (!calls[node]) continue;
        PyJitNodeRef *h=&nodes[node]->handle;
        PyObject *expected[3]={h->physical_callable?h->physical_callable:h->callable,h->receiver,h->keyword_names};
        for (int role=0;role<2+keywords[node];role++) {
            int value=call_values[node][role], source=INT16_MIN;
            for (int slot=0;slot<physical_count;slot++) {
                if (physical_inputs[slot]==value) {source=slot;break;}
            }
            if (source==INT16_MIN) {
                int proven_constant=0;
                for (int known=0;known<known_load_count;known++) {
                    if (load_runtime_values[known]!=value) continue;
                    if (load_locals[known]>=0 && load_locals[known]<INT16_MAX) source=-load_locals[known]-1;
                    else if (expected[role] && load_constants[known]==expected[role]) proven_constant=1;
                    else if (role==1 && expected[role]==NULL && load_locals[known]==-2) proven_constant=1;
                    break;
                }
                if (proven_constant) continue;
                if (source==INT16_MIN) return crossjit_reject("call root has no guarded entry source");
            }
            call_sources[call_count]=source;
            call_objects[call_count++]=expected[role];
        }
    }

    /* Complete the physical portion of Pi_H only after stack simulation has
     * proved the mapping.  No Provider callback can influence these fields. */
    region->proof.deopt_uop_index = (uint32_t)first_set_ip;
    region->proof.rewrite_first = (uint32_t)first_set_ip;
    region->proof.rewrite_last = (uint32_t)last;
    memcpy(region->proof.input_sources, input_sources,
           region->input_count * sizeof(input_sources[0]));
    _PyInterpreterState_GET()->crossjit_stats.physical_proof_validated++;

    PyJitRegionView view = {region, state};
    CrossJitExecutorPlan lowered;
    PyJitPrepareDecision decision;
    int lower_status = crossjit_lower_region(&view, &lowered, &decision);
    if (lower_status <= 0) {
        if (lower_status < 0) {
            return lower_status;
        }
        crossjit_reject("provider lowering declined the proven region");
        /* Host's outer bounded search independently proves closed smaller
         * regions. A non-closed immediate prefix must not mask already
         * admitted paths, and recursive prefix retries would duplicate work. */
        state->last_lowering_decline=decision;
        return 0;
    }
    lowered.stack_input_count = physical_count;
    lowered.call_guard_count=call_count;
    for (int i=0;i<call_count;i++) {
        lowered.call_guard_sources[i]=call_sources[i];
        lowered.call_guard_objects[i]=Py_XNewRef(call_objects[i]);
    }
    lowered.empty_local_count = (uint8_t)transient_count;
    memcpy(lowered.empty_locals, transient_locals,
           (size_t)transient_count * sizeof(transient_locals[0]));
    lowered.type_guard_count = (uint8_t)guard_count;
    memcpy(lowered.type_guard_inputs, guard_inputs, (size_t)guard_count);
    memcpy(lowered.type_guard_types, guard_types,
           (size_t)guard_count * sizeof(guard_types[0]));
    memcpy(lowered.input_sources, input_sources,
           region->input_count * sizeof(input_sources[0]));
    /* Pin only proven original inline constants, after successful lowering.
     * No object is evaluated or synthesized, and MISS replays original loads.
     * Published plans expose these references to cyclic GC and release them
     * on invalidation; every unpublished-plan failure releases them too. */
    for (uint8_t input = 0; input < region->input_count; input++) {
        lowered.input_constants[input] = Py_XNewRef(input_constants[input]);
    }
    CrossJitExecutorPlan *pending = PyMem_Malloc(sizeof(*pending));
    if (pending == NULL) {
        crossjit_destroy_unpublished_plan(&lowered);
        PyErr_NoMemory();
        return -1;
    }
    *pending = lowered;

    _PyUOpInstruction original_set_ip = buffer[first_set_ip];
    _PyUOpInstruction original_exec = buffer[first_exec];
    int validity_count=0;
    for (int i = first_exec + 1; i < last_exec; i++) {
        uint16_t opcode = crossjit_base_opcode(&buffer[i]);
        validity_count+=opcode==_CHECK_VALIDITY;
    }
    int resume_check=0;
    for (int i=first_exec+1;i<last_exec;i++) resume_check|=crossjit_base_opcode(&buffer[i])==_TIER2_RESUME_CHECK;
    int separate_guard = pending->guard_plan != NULL && !pending->combined_guards;
    int needed_slots = 1 + resume_check + separate_guard + 1;
    if (first_set_ip + validity_count + needed_slots > last + 1) {
        PyMem_Free(pending);
        crossjit_destroy_unpublished_plan(&lowered);
        return crossjit_reject("fixed-width rewrite has insufficient slots");
    }
    /* All declining attempts leave the trace unchanged, allowing Host to
     * prove a smaller closed region without inheriting a partial rewrite. */
    int write=first_set_ip;
    for (int i=first_exec+1;i<last_exec;i++) {
        if (crossjit_base_opcode(&buffer[i])==_CHECK_VALIDITY) buffer[write++]=buffer[i];
    }
    buffer[write++] = original_set_ip;
    if (resume_check) {
        /* Poll before the atomic, GIL-held provider region. A pending event
         * deopts with the original entry stack and instruction pointer; the
         * final call's periodic check remains after successful publication. */
        _PyUOpInstruction check=original_exec;
        check.opcode=_TIER2_RESUME_CHECK;
        check.oparg=0;
        check.operand0=0;
        buffer[write++]=check;
    }
    if (separate_guard) {
        _PyUOpInstruction guard = original_exec;
        guard.opcode = _EXT_GUARD;
        guard.oparg = pending->stack_input_count;
        guard.operand0 = (uintptr_t)pending;
        buffer[write++] = guard;
    }
    _PyUOpInstruction execute = original_exec;
    execute.opcode = _EXT_REGION_EXEC;
    execute.oparg = pending->stack_input_count;
    execute.operand0 = (uintptr_t)pending;
    buffer[write++] = execute;
    for (int clear = write; clear <= last; clear++) {
        crossjit_make_nop(&buffer[clear]);
    }
    if (transient_count) {
        _PyInterpreterState_GET()->crossjit_stats.transient_local_regions++;
    }
    if (guard_count) {
        _PyInterpreterState_GET()->crossjit_stats.type_guard_regions++;
    }
    return 1;
}

int
_PyCrossJit_RewriteRegions(CrossJitAnalysisState *state,
                           _PyUOpInstruction *buffer, int length)
{
    /* Rewrites independent accepted regions after all semantic proofs finish. */
    if (state == NULL || buffer == NULL || length <= 0) {
        return 0;
    }
    int rewritten = 0;
    for (size_t i = 0; i < state->region_count; i++) {
        CrossJitRegion *candidate = &state->regions[i];
        uint64_t validated_before=
            _PyInterpreterState_GET()->crossjit_stats.physical_proof_validated;
        int status = crossjit_rewrite_region(state, candidate, buffer, length);
        /* A broad closed DAG may fail physical proof at a local boundary.
         * Host considers bounded closed subregions, including a single call,
         * and independently proves each one. No visible write is crossed. */
        int minimum=PROVIDER_HAS(candidate->provider,supports_single_node_regions) &&
            candidate->provider->supports_single_node_regions?1:2;
        int may_retry=_PyInterpreterState_GET()->crossjit_stats.physical_proof_validated==validated_before ||
            state->last_lowering_decline==PYJIT_PREPARE_UNSUPPORTED_NODE ||
            state->last_lowering_decline==PYJIT_PREPARE_UNPROFITABLE;
        for (int count=candidate->node_count-1;status==0 && count>=minimum && may_retry;count--) {
          for (int skip=0;status==0 && skip+count<=candidate->node_count && may_retry;skip++) {
            CrossJitRegion suffix = {
                .provider = candidate->provider,
                .first_node = candidate->first_node + skip,
                .node_count = (uint8_t)count,
                .result_count = 1,
            };
            CrossJitNode *tail=&state->nodes[suffix.first_node+count-1];
            suffix.results[0]=(PyJitRegionValue){value_for_id(state,tail->results[0]),1,1};
            suffix.proof.node_count = suffix.node_count;
            suffix.proof.result_count = 1;
            suffix.proof.virtual_value_count = suffix.node_count - 1;
            if (!suffix.results[0].value || (count==1 && !node_is_virtualizable(&state->nodes[suffix.first_node])) ||
                !region_is_closed_dag(state, suffix.first_node,
                                      suffix.first_node + suffix.node_count) ||
                !region_collect_inputs(state, &suffix)) {
                continue;
            }
            suffix.proof.live_in_count = suffix.input_count;
            if (host_proof_cache_observe(state, &suffix) < 0) {
                return -1;
            }
            status = crossjit_rewrite_region(state, &suffix, buffer, length);
            if (state->last_lowering_decline==PYJIT_PREPARE_UNSUPPORTED_CONTRACT)
                may_retry=0;
          }
        }
        if (status < 0) {
            return -1;
        }
        rewritten += status;
        if (status > 0) {
            _PyInterpreterState_GET()->crossjit_stats.rewritten_regions++;
        }
    }
    return rewritten;
}

/* Returns the unpublished plan pointer embedded in a generic CrossJIT uop. */
static CrossJitExecutorPlan *
crossjit_plan_from_uop(const _PyUOpInstruction *instruction)
{
    uint16_t opcode = crossjit_base_opcode(instruction);
    if ((opcode == _EXT_GUARD || opcode == _EXT_REGION_EXEC) &&
        instruction->operand0 != 0) {
        return (CrossJitExecutorPlan *)(uintptr_t)instruction->operand0;
    }
    return NULL;
}

/* Returns whether the same unpublished plan appeared in an earlier uop. */
static int
crossjit_plan_seen(_PyUOpInstruction *buffer, int index,
                   CrossJitExecutorPlan *plan)
{
    for (int i = 0; i < index; i++) {
        if (crossjit_plan_from_uop(&buffer[i]) == plan) {
            return 1;
        }
    }
    return 0;
}

void
_PyCrossJit_FreePlansFromUops(_PyUOpInstruction *buffer, int length)
{
    /* Releases plans when executor allocation fails before ownership transfer. */
    for (int i = 0; i < length; i++) {
        CrossJitExecutorPlan *plan = crossjit_plan_from_uop(&buffer[i]);
        if (plan != NULL && !crossjit_plan_seen(buffer, i, plan)) {
            crossjit_destroy_unpublished_plan(plan);
            PyMem_Free(plan);
        }
    }
}

int
_PyCrossJit_ExecutorPlansInit(_PyExecutorObject *executor,
                              _PyUOpInstruction *buffer, int length)
{
    /* Copies unique pending plans and replaces raw pointers with table indices. */
    uint32_t count = 0;
    for (int i = 0; i < length; i++) {
        CrossJitExecutorPlan *plan = crossjit_plan_from_uop(&buffer[i]);
        if (plan != NULL && !crossjit_plan_seen(buffer, i, plan)) {
            count++;
        }
    }
    if (count == 0) {
        return 0;
    }
    if (count > UINT16_MAX) {
        _PyCrossJit_FreePlansFromUops(buffer, length);
        PyErr_SetString(PyExc_OverflowError, "too many CrossJIT executor plans");
        return -1;
    }
    CrossJitExecutorPlan *plans = PyMem_Calloc(count, sizeof(*plans));
    CrossJitExecutorPlan **pending = PyMem_Malloc(count * sizeof(*pending));
    if (plans == NULL || pending == NULL) {
        PyMem_Free(plans);
        PyMem_Free(pending);
        _PyCrossJit_FreePlansFromUops(buffer, length);
        PyErr_NoMemory();
        return -1;
    }
    uint16_t next = 0;
    for (int i = 0; i < length; i++) {
        CrossJitExecutorPlan *plan = crossjit_plan_from_uop(&buffer[i]);
        if (plan != NULL && !crossjit_plan_seen(buffer, i, plan)) {
            plans[next] = *plan;
            pending[next] = plan;
            next++;
        }
    }
    for (int i = 0; i < length; i++) {
        CrossJitExecutorPlan *plan = crossjit_plan_from_uop(&buffer[i]);
        if (plan == NULL) {
            continue;
        }
        uint16_t index = 0;
        while (index < next && pending[index] != plan) {
            index++;
        }
        assert(index < next);
        buffer[i].operand0 = index;
    }
    for (uint16_t i = 0; i < next; i++) {
        PyMem_Free(pending[i]);
    }
    PyMem_Free(pending);
    executor->crossjit_plans = plans;
    executor->crossjit_plan_count = next;
    _PyInterpreterState_GET()->crossjit_stats.published_regions += next;
    return 0;
}

/* Public read-only semantic and proof accessors. */
PyAPI_FUNC(PyObject *)
PyJit_ValueGetObservedObject(const PyJitValueRef *value)
{
    return value == NULL ? NULL : value->observed;
}

PyAPI_FUNC(PyTypeObject *)
PyJit_ValueGetExactType(const PyJitValueRef *value)
{
    return value == NULL ? NULL : value->exact_type;
}

PyAPI_FUNC(uint32_t)
PyJit_ValueGetId(const PyJitValueRef *value)
{
    return value == NULL ? 0 : value->id;
}

PyAPI_FUNC(const PyJitValueRef *)
PyJit_NodeGetOperand(const PyJitNodeRef *node, uint8_t index)
{
    return node != NULL && index < node->operand_count ?
        node->operands[index] : NULL;
}

PyAPI_FUNC(const PyJitValueRef *)
PyJit_NodeGetResult(const PyJitNodeRef *node, uint8_t index)
{
    return node != NULL && index < node->result_count ?
        node->results[index] : NULL;
}

PyAPI_FUNC(uint8_t)
PyJit_NodeGetOperandCount(const PyJitNodeRef *node)
{
    return node == NULL ? 0 : node->operand_count;
}

PyAPI_FUNC(PyObject *)
PyJit_NodeGetKeywordNames(const PyJitNodeRef *node)
{
    return node == NULL ? NULL : node->keyword_names;
}

PyAPI_FUNC(uint8_t)
PyJit_NodeGetPositionalCount(const PyJitNodeRef *node)
{
    return node == NULL ? 0 : node->positional_count;
}

PyAPI_FUNC(uint8_t)
PyJit_NodeGetCallBindingKind(const PyJitNodeRef *node)
{
    return node == NULL ? 0 : node->call_binding_kind;
}

PyAPI_FUNC(uint8_t)
PyJit_NodeGetResultCount(const PyJitNodeRef *node)
{
    return node == NULL ? 0 : node->result_count;
}

PyAPI_FUNC(uint64_t)
PyJit_NodeGetSemanticId(const PyJitNodeRef *node)
{
    return node == NULL ? 0 : node->semantic_id;
}

PyAPI_FUNC(const PyJitSemanticSummary *)
PyJit_NodeGetSemanticSummary(const PyJitNodeRef *node)
{
    return node == NULL ? NULL : &node->summary;
}

PyAPI_FUNC(uint8_t)
PyJit_RegionGetNodeCount(const PyJitRegionView *region)
{
    return region == NULL || region->region == NULL ?
        0 : region->region->node_count;
}

PyAPI_FUNC(const PyJitNodeRef *)
PyJit_RegionGetNode(const PyJitRegionView *region, uint8_t index)
{
    if (region == NULL || region->region == NULL || region->state == NULL ||
        index >= region->region->node_count) {
        return NULL;
    }
    return &region->state->nodes[
        region->region->first_node + index].handle;
}

PyAPI_FUNC(uint8_t)
PyJit_RegionGetInputCount(const PyJitRegionView *region)
{
    return region == NULL || region->region == NULL ?
        0 : region->region->input_count;
}

PyAPI_FUNC(const PyJitRegionValue *)
PyJit_RegionGetInput(const PyJitRegionView *region, uint8_t index)
{
    if (region == NULL || region->region == NULL ||
        index >= region->region->input_count) {
        return NULL;
    }
    return &region->region->inputs[index];
}

PyAPI_FUNC(uint8_t)
PyJit_RegionGetResultCount(const PyJitRegionView *region)
{
    return region == NULL || region->region == NULL ?
        0 : region->region->result_count;
}

PyAPI_FUNC(const PyJitRegionValue *)
PyJit_RegionGetResult(const PyJitRegionView *region, uint8_t index)
{
    if (region == NULL || region->region == NULL ||
        index >= region->region->result_count) {
        return NULL;
    }
    return &region->region->results[index];
}

PyAPI_FUNC(uint64_t)
PyJit_RegionGetStructuralFingerprint(const PyJitRegionView *region)
{
    return region == NULL || region->region == NULL ?
        0 : region->region->proof.structural_fingerprint;
}
