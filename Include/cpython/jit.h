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
    PYJIT_EXT_TRACE_FUSED_BINARY2,
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
        struct {
            PyJitExtPlan *first_plan;
            PyJitExtPlan *second_plan;
            uint32_t flags;
        } fused_binary2;
    } u;
} PyJitExtTraceRequest;

#define PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_LEFT   (1u << 0)
#define PYJIT_EXT_FUSED_BINARY2_SECOND_CONSUMES_RIGHT  (1u << 1)

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *op_name;
    const char *layout_name;
    const char *left_role;
    const char *right_role;
    const char *result_role;
    const char *left_dtype;
    const char *right_dtype;
    const char *result_dtype;
    int itemsize;
    int64_t min_elements;
    const char *fused_op_name;
    const char *fused_pattern;
} PyJitExtPlanInfo;

struct _PyJitExtProvider {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *provider_name;
    uint32_t provider_flags;
    PyJitExtPlan *(*try_plan)(
        const PyJitExtTraceRequest *request,
        PyJitExtSiteState *site);
    void (*free_plan)(PyJitExtPlan *plan);
    int (*describe_plan)(
        const PyJitExtPlan *plan,
        PyJitExtPlanInfo *info);
};

typedef enum {
    PYJIT_EXT_PLAN_BINARY,
    PYJIT_EXT_PLAN_CALL,
    PYJIT_EXT_PLAN_FUSED_BINARY2,
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

typedef PyJitExtResult (*PyJitExtFusedBinary2EvalFunc)(
    void *payload,
    PyObject *first_left,
    PyObject *first_right,
    PyObject *second_other,
    PyObject **result);

typedef enum {
    PYJIT_EXT_STENCIL_PATCH_ABS64,
    PYJIT_EXT_STENCIL_PATCH_REL32,
} PyJitExtStencilPatchKind;

typedef enum {
    PYJIT_EXT_STENCIL_VALUE_PLAN,
    PYJIT_EXT_STENCIL_VALUE_PAYLOAD,
    PYJIT_EXT_STENCIL_VALUE_BINARY_EVAL,
    PYJIT_EXT_STENCIL_VALUE_CALL_EVAL,
    PYJIT_EXT_STENCIL_VALUE_FUSED_BINARY2_EVAL,
    PYJIT_EXT_STENCIL_VALUE_NEXT_INSTRUCTION,
    PYJIT_EXT_STENCIL_VALUE_TARGET,
    PYJIT_EXT_STENCIL_VALUE_ERROR_TARGET,
    PYJIT_EXT_STENCIL_VALUE_RECORD_EXT_HIT,
    PYJIT_EXT_STENCIL_VALUE_RECORD_EXT_MISS,
} PyJitExtStencilValueKind;

typedef enum {
    PYJIT_EXT_STENCIL_SECTION_CODE,
    PYJIT_EXT_STENCIL_SECTION_DATA,
} PyJitExtStencilSection;

typedef struct {
    uint32_t offset;
    uint16_t section;
    uint16_t patch_kind;
    uint16_t value_kind;
    uint16_t reserved;
    int64_t addend;
} PyJitExtStencilHole;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const unsigned char *code;
    size_t code_size;
    const unsigned char *data;
    size_t data_size;
    const PyJitExtStencilHole *holes;
    size_t hole_count;
} PyJitExtStencil;

typedef const PyJitExtStencil *(*PyJitExtGetStencilFunc)(
    const PyJitExtPlan *plan,
    int opcode);

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
        PyJitExtFusedBinary2EvalFunc fused_binary2_eval;
    } eval;
    void (*free_payload)(void *payload);
    const char *debug_name;
    PyJitExtSiteState *site;
    const PyJitExtStencil *stencil;
    PyJitExtGetStencilFunc get_stencil;
};

#define PYJIT_EXT_PLAN_REQUIRES_GIL          (1u << 0)
#define PYJIT_EXT_PLAN_MAY_RELEASE_GIL       (1u << 1)
#define PYJIT_EXT_PLAN_NO_PYTHON_CALLBACK    (1u << 2)
#define PYJIT_EXT_PLAN_ALLOCATES_RESULT      (1u << 3)
#define PYJIT_EXT_PLAN_DEVICE_ASYNC          (1u << 4)
#define PYJIT_EXT_PLAN_NEEDS_SYNC_ON_DEOPT   (1u << 5)
#define PYJIT_EXT_PLAN_OPAQUE_EFFECTS        (1u << 6)

#define PYJIT_EXT_GUARD_FAIL_REASON_COUNT 9
#define PYJIT_EXT_BINARY_PROVIDER_CACHE_MAX 64
#define PYJIT_EXT_CALLABLE_PROVIDER_CACHE_MAX 16

typedef enum {
    PYJIT_EXT_OPT_VISIT,
    PYJIT_EXT_OPT_SUPPORTED_OP_SKIP,
    PYJIT_EXT_OPT_PROVIDER_PREFILTER_MISS,
    PYJIT_EXT_OPT_PROVIDER_LOOKUP,
    PYJIT_EXT_OPT_TRY_PLAN,
    PYJIT_EXT_OPT_PLAN_SUCCESS,
    PYJIT_EXT_OPT_EMIT_UOP,
    PYJIT_EXT_OPT_EXECUTOR_CREATED,
    PYJIT_EXT_OPT_EXECUTOR_INVALIDATED,
    PYJIT_EXT_OPT_EVENT_COUNT,
} PyJitExtOptimizerEvent;

typedef enum {
    PYJIT_EXT_BINARY_ADMISSION_VISIT,
    PYJIT_EXT_BINARY_ADMISSION_SUPPORTED_OP_SKIP,
    PYJIT_EXT_BINARY_ADMISSION_MISSING_LEFT_TYPE,
    PYJIT_EXT_BINARY_ADMISSION_MISSING_RIGHT_TYPE,
    PYJIT_EXT_BINARY_ADMISSION_UNSUPPORTED_TYPE_PAIR,
    PYJIT_EXT_BINARY_ADMISSION_PROVIDER_NOT_FOUND,
    PYJIT_EXT_BINARY_ADMISSION_SITE_DISABLED,
    PYJIT_EXT_BINARY_ADMISSION_SITE_DISABLED_NUMPY_SHAPE_MISMATCH,
    PYJIT_EXT_BINARY_ADMISSION_SITE_DISABLED_NUMPY_SCALAR_ARRAY_NOT_C_CONTIGUOUS,
    PYJIT_EXT_BINARY_ADMISSION_SITE_DISABLED_MISS_THRESHOLD,
    PYJIT_EXT_BINARY_ADMISSION_SITE_DISABLED_STRIDED_MIN_ELEMENTS,
    PYJIT_EXT_BINARY_ADMISSION_TRY_PLAN,
    PYJIT_EXT_BINARY_ADMISSION_TRY_PLAN_NULL,
    PYJIT_EXT_BINARY_ADMISSION_PLAN_SUCCESS,
    PYJIT_EXT_BINARY_ADMISSION_EMIT_EXTERNAL_UOP,
    PYJIT_EXT_BINARY_ADMISSION_REASON_COUNT,
} PyJitExtBinaryAdmissionReason;

typedef struct _PyJitExtBinaryAdmissionEntry {
    int binary_oparg;
    PyJitExtBinaryAdmissionReason reason;
    PyTypeObject *left_type;
    PyTypeObject *right_type;
    char left_type_name[96];
    char right_type_name[96];
    uint64_t count;
    struct _PyJitExtBinaryAdmissionEntry *next;
} PyJitExtBinaryAdmissionEntry;

struct _PyJitExtSiteState {
    uint16_t hit_count;
    uint16_t miss_count;
    uint16_t decline_count;
    uint8_t disabled;
    uint8_t retry_counter;
    uint8_t trace_kind;
    uint32_t last_plan_op_id;
    uint32_t site_id;
    PyJitExtProvider *last_provider;
    const char *disable_reason;
    uint64_t optimizer_skip_after_disable;
    uint64_t guard_fail_counts[PYJIT_EXT_GUARD_FAIL_REASON_COUNT];
    int64_t element_count_min;
    int64_t element_count_max;
    int64_t element_count_last;
    uint16_t fused_site_count;
    uint16_t fused_hit_count;
    uint16_t fused_miss_count;
    const char *fused_pattern;
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
