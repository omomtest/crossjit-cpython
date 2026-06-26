#ifndef Py_CPYTHON_JIT_H
#define Py_CPYTHON_JIT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
 * Experimental external JIT plan API (non-limited C API).
 *
 * CPython owns provider lookup and executor lifetime. Extension modules own
 * plan payloads and run their own guards, allocation, and kernels from eval().
 */

#define PYJIT_EXT_ABI_VERSION 1u

typedef enum {
    PYJIT_EXT_HIT = 1,
    PYJIT_EXT_MISS = 0,
    PYJIT_EXT_ERROR = -1,
} PyJitExtResult;

typedef enum {
    PYJIT_EXT_TRACE_BINARY,
    PYJIT_EXT_TRACE_CALL,
} PyJitExtTraceKind;

typedef struct _PyJitExtProvider PyJitExtProvider;
typedef struct _PyJitExtPlan PyJitExtPlan;
typedef struct _PyJitExtSiteState PyJitExtSiteState;

typedef struct {
    PyJitExtTraceKind kind;
    union {
        struct {
            int binary_oparg;
            PyTypeObject *left_type;
            PyTypeObject *right_type;
        } binary;
        struct {
            PyObject *callable;
            Py_ssize_t nargs;
            uint32_t call_flags;
            PyObject *kwnames;
        } call;
    } u;
} PyJitExtTraceRequest;

struct _PyJitExtProvider {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *provider_name;
    uint32_t provider_flags;
    PyJitExtPlan *(*try_plan)(
        const PyJitExtTraceRequest *request,
        PyJitExtSiteState *site);
    void (*free_plan)(PyJitExtPlan *plan);
};

typedef enum {
    PYJIT_EXT_PLAN_BINARY,
    PYJIT_EXT_PLAN_CALL,
} PyJitExtPlanKind;

typedef PyJitExtResult (*PyJitExtBinaryEvalFunc)(
    void *payload,
    PyObject *left,
    PyObject *right,
    PyObject **result);

typedef PyJitExtResult (*PyJitExtCallEvalFunc)(
    void *payload,
    PyObject *callable,
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject **result);

struct _PyJitExtPlan {
    uint32_t abi_version;
    uint32_t struct_size;
    PyJitExtPlanKind kind;
    PyJitExtProvider *provider;
    uint32_t flags;
    uint32_t op_id;
    void *payload;
    union {
        PyJitExtBinaryEvalFunc binary_eval;
        PyJitExtCallEvalFunc call_eval;
    } eval;
    void (*free_payload)(void *payload);
    const char *debug_name;
};

#define PYJIT_EXT_PLAN_REQUIRES_GIL          (1u << 0)
#define PYJIT_EXT_PLAN_MAY_RELEASE_GIL       (1u << 1)
#define PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK    (1u << 2)
#define PYJIT_EXT_PLAN_ALLOCATES_RESULT      (1u << 3)
#define PYJIT_EXT_PLAN_DEVICE_ASYNC          (1u << 4)
#define PYJIT_EXT_PLAN_NEEDS_SYNC_ON_DEOPT   (1u << 5)
#define PYJIT_EXT_PLAN_OPAQUE_EFFECTS        (1u << 6)

struct _PyJitExtSiteState {
    uint16_t hit_count;
    uint16_t miss_count;
    uint16_t decline_count;
    uint8_t disabled;
    uint8_t retry_counter;
    uint32_t last_plan_op_id;
    PyJitExtProvider *last_provider;
};

#define PYJIT_CALL_HAS_KWARGS        (1u << 0)
#define PYJIT_CALL_HAS_VARARGS       (1u << 1)

PyAPI_FUNC(int) PyJit_RegisterBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg,
    PyJitExtProvider *provider);
PyAPI_FUNC(int) PyJit_UnregisterBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg);
PyAPI_FUNC(PyJitExtProvider *) PyJit_LookupBinaryProvider(
    PyTypeObject *left_type,
    PyTypeObject *right_type,
    int binary_oparg);

PyAPI_FUNC(int) PyJit_RegisterCallableProvider(
    PyObject *callable,
    PyJitExtProvider *provider);
PyAPI_FUNC(int) PyJit_UnregisterCallableProvider(PyObject *callable);
PyAPI_FUNC(PyJitExtProvider *) PyJit_LookupCallableProvider(PyObject *callable);

#ifdef __cplusplus
}
#endif
#endif /* !Py_CPYTHON_JIT_H */
