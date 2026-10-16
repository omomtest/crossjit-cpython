#ifndef Py_INTERNAL_CROSSJIT_H
#define Py_INTERNAL_CROSSJIT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "cpython/jit.h"
#include "pycore_interpframe_structs.h" /* _PyInterpreterFrame */
#include "pycore_uop.h"          /* _PyUOpInstruction */

typedef struct CrossJitAnalysisState CrossJitAnalysisState;

/* Owns one provider plan after a trace is published as an executor. */
typedef struct {
    const PyJitProviderSpec *provider; /* borrowed static provider descriptor */
    PyJitGuardPlan *guard_plan;  /* owned; optional and destroyed exactly once */
    PyJitExecPlan *exec_plan;    /* owned; required for executable regions */
    uint8_t input_count;          /* semantic provider inputs */
    uint8_t stack_input_count;    /* inputs consumed from the value stack */
    uint8_t combined_guards;      /* optional guard/execute callback at entry */
    uint8_t empty_local_count;    /* Host-proven transient locals, NULL at entry/exit */
    int16_t empty_locals[PYJIT_REGION_MAX_NODES];
    /* Exact built-in type tests reconstructed from stack-preserving Host
     * guards. Dependencies are canonical live-ins, never virtual outputs. */
    uint8_t call_guard_count;
    int16_t call_guard_sources[PYJIT_REGION_MAX_NODES * 3];
    PyObject *call_guard_objects[PYJIT_REGION_MAX_NODES * 3]; /* owned; NULL means NULL self */
    uint8_t type_guard_count;
    uint8_t type_guard_inputs[PYJIT_REGION_MAX_NODES * 2];
    PyTypeObject *type_guard_types[PYJIT_REGION_MAX_NODES * 2];
    /* Non-negative entries index the consumed stack slice. Negative entries
     * encode local N as -(N + 1), except INT16_MIN for an owned constant. */
    int16_t input_sources[PYJIT_REGION_MAX_LIVE_INS];
    PyObject *input_constants[PYJIT_REGION_MAX_LIVE_INS]; /* owned or NULL */
} CrossJitExecutorPlan;

struct _PyExecutorObject;

/* Kept apart from JitOptSymbol so a normal Tier-2 trace pays nothing. */
CrossJitAnalysisState *_PyCrossJit_AnalysisNew(void);
void _PyCrossJit_AnalysisClear(CrossJitAnalysisState *state);
void _PyCrossJit_AnalysisFree(CrossJitAnalysisState *state);

int _PyCrossJit_IsEnabled(void);
/* True when providers should run abstract evaluation without region lowering. */
int _PyCrossJit_IsSemanticsOnly(void);
const PyJitProviderSpec *_PyCrossJit_LookupUnaryProvider(
    PyInterpreterState *interp, PyTypeObject *operand, int operation);
const PyJitProviderSpec *_PyCrossJit_LookupCompareProvider(
    PyInterpreterState *interp, PyTypeObject *left, PyTypeObject *right,
    int operation);
const PyJitProviderSpec *_PyCrossJit_LookupCallableProvider(
    PyInterpreterState *interp, PyObject *callable);
void _PyCrossJit_InterpreterClear(PyInterpreterState *interp);
PyAPI_FUNC(PyObject *) _PyCrossJit_GetStats(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyCrossJit_ResetStats(PyInterpreterState *interp);

/* Forms maximal same-provider candidates after abstract interpretation. */
int _PyCrossJit_FormRegions(CrossJitAnalysisState *state,
                            _PyUOpInstruction *buffer, int length);
int _PyCrossJit_RewriteRegions(CrossJitAnalysisState *state,
                               _PyUOpInstruction *buffer, int length);

/* Releases every plan exactly once when an executor is invalidated or freed. */
void _PyCrossJit_ExecutorPlansClear(struct _PyExecutorObject *executor);
int _PyCrossJit_ExecutorPlansTraverse(struct _PyExecutorObject *executor,
                                     visitproc visit, void *arg);
int _PyCrossJit_ExecutorPlansInit(struct _PyExecutorObject *executor,
                                  _PyUOpInstruction *buffer, int length);
void _PyCrossJit_FreePlansFromUops(_PyUOpInstruction *buffer, int length);

/* Dispatches provider-owned plans while preserving Host stack ownership. */
PyJitResult _PyCrossJit_ExecutorRunGuards(
    struct _PyExecutorObject *executor, uint32_t plan_index,
    PyObject *const *inputs, Py_ssize_t input_count);
PyJitResult _PyCrossJit_ExecutorRunRegion(
    struct _PyExecutorObject *executor, uint32_t plan_index,
    PyObject *const *inputs, Py_ssize_t input_count, PyObject **result);
int _PyCrossJit_ExecutorPrepareInputs(
    struct _PyExecutorObject *executor, uint32_t plan_index,
    _PyInterpreterFrame *frame, _PyStackRef *stack_inputs,
    Py_ssize_t stack_input_count, PyObject **inputs, Py_ssize_t capacity);

#ifdef __cplusplus
}
#endif
#endif /* Py_INTERNAL_CROSSJIT_H */
