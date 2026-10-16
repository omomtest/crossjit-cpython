#ifndef Py_CPYTHON_JIT_H
#define Py_CPYTHON_JIT_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Experimental CrossJIT provider API.
 *
 * Providers project private domain state D_P into a small, stable Host
 * semantic vocabulary S_H.  CPython forms and proves regions exclusively
 * from S_H plus Host program facts.  Only after that proof is complete does
 * CPython ask the provider to construct its opaque specialization contract
 * C_P and execution plan Pi_P.
 *
 * Optimizer IR, executor layout, deoptimization state and uop storage are
 * deliberately absent from this versioned, non-limited API.
 */
#ifndef Py_LIMITED_API

#include <stddef.h>
#include <stdint.h>

#define PYJIT_PROVIDER_ABI_VERSION 4u
#define PYJIT_CALL_DESCRIPTOR_VERSION 1u
#define PYJIT_SINGLE_NODE_REGION_CAPABILITY 1
#define PYJIT_REGION_MAX_NODES 12u
#define PYJIT_REGION_MAX_LIVE_INS 16u
#define PYJIT_OPERATION_MAX_OPERANDS 8u
#define PYJIT_OPERATION_MAX_RESULTS 4u
#define PYJIT_UNARY_NEGATIVE 1

typedef struct PyJitValueRef PyJitValueRef;
typedef struct PyJitNodeRef PyJitNodeRef;
typedef struct PyJitRegionView PyJitRegionView;
typedef struct PyJitGuardPlan PyJitGuardPlan;
typedef struct PyJitExecPlan PyJitExecPlan;

typedef enum {
    PYJIT_RESULT_ERROR = -1,
    PYJIT_RESULT_MISS = 0,
    PYJIT_RESULT_HIT = 1,
} PyJitResult;

typedef enum {
    PYJIT_OPERATION_BINARY = 1,
    PYJIT_OPERATION_UNARY = 2,
    PYJIT_OPERATION_CALL = 3,
    PYJIT_OPERATION_SUBSCRIPT = 4,
    PYJIT_OPERATION_COMPARE = 5,
} PyJitOperationKind;

typedef enum {
    PYJIT_KEY_BINARY = 1,
    PYJIT_KEY_UNARY = 2,
    PYJIT_KEY_CALLABLE = 3,
    PYJIT_KEY_COMPARE = 4,
} PyJitProviderKeyKind;

typedef enum {
    PYJIT_MUTATION_NONE = 1,
    PYJIT_MUTATION_ARGUMENT = 2,
    PYJIT_MUTATION_GLOBAL = 3,
    PYJIT_MUTATION_UNKNOWN = 4,
} PyJitMutationClass;

typedef enum {
    PYJIT_ALIAS_FRESH = 1,
    PYJIT_ALIAS_MAY_ALIAS_ARGUMENT = 2,
    PYJIT_ALIAS_UNKNOWN = 3,
} PyJitAliasClass;

typedef enum {
    PYJIT_MATERIALIZATION_VIRTUALIZABLE = 1,
    PYJIT_MATERIALIZATION_REQUIRED = 2,
    PYJIT_MATERIALIZATION_UNKNOWN = 3,
} PyJitMaterializationClass;

typedef enum {
    PYJIT_REPLAYABLE = 1,
    PYJIT_NOT_REPLAYABLE = 2,
    PYJIT_REPLAYABILITY_UNKNOWN = 3,
} PyJitReplayability;

/* Host-visible properties that may be justified by Provider assumptions.
 * A Provider sets these bits on each PyJitAssumption whose runtime validity
 * is required for the corresponding projected property.  Several properties
 * may share one assumption and one property may require several assumptions.
 * Zero is valid for assumptions used only by the private specialization C_P. */
typedef enum {
    PYJIT_SEMANTIC_PROPERTY_MUTATION = UINT64_C(1) << 0,
    PYJIT_SEMANTIC_PROPERTY_RESULT_ALIAS = UINT64_C(1) << 1,
    PYJIT_SEMANTIC_PROPERTY_MATERIALIZATION = UINT64_C(1) << 2,
    PYJIT_SEMANTIC_PROPERTY_REPLAYABILITY = UINT64_C(1) << 3,
    PYJIT_SEMANTIC_PROPERTY_MAY_RAISE = UINT64_C(1) << 4,
    PYJIT_SEMANTIC_PROPERTY_MAY_CALL_HOST = UINT64_C(1) << 5,
    PYJIT_SEMANTIC_PROPERTY_MAY_RELEASE_GIL = UINT64_C(1) << 6,
    PYJIT_SEMANTIC_PROPERTY_EXACT_RESULT_TYPE = UINT64_C(1) << 7,
} PyJitSemanticProperty;

#define PYJIT_SEMANTIC_PROPERTY_MASK \
    ((UINT64_C(1) << 8) - UINT64_C(1))

/* S_H: the complete Provider-to-Host semantic projection.  All reserved
 * fields must be zero.  CPython normalizes this value before semantic-class
 * comparison; trace-specific physical proof is never cached. */
typedef struct {
    uint32_t struct_size;
    PyJitMutationClass mutation;
    PyJitAliasClass result_alias;
    PyJitMaterializationClass materialization;
    PyJitReplayability replayability;
    uint8_t may_raise;
    uint8_t may_call_host;
    uint8_t may_release_gil;
    uint8_t reserved8;
    PyTypeObject *exact_result_type; /* borrowed exact Host type, or NULL */
    uint64_t reserved[2];
} PyJitSemanticSummary;

typedef struct {
    PyJitProviderKeyKind kind;
    int operation;
    PyTypeObject *left_type;
    PyTypeObject *right_type;
    PyObject *callable;
} PyJitProviderKey;

typedef struct {
    PyJitOperationKind kind;
    int operation;
    PyObject *callable; /* borrowed, only for CALL */
    PyJitValueRef *const *operands; /* borrowed Host semantic identities */
    uint8_t operand_count;
    uint8_t result_count;
    uint8_t has_keywords;
    uint8_t reserved;
    uint32_t call_descriptor_version;
    uint8_t positional_count;
    PyObject *keyword_names; /* borrowed exact tuple, names align with trailing operands */
} PyJitRecognizeRequest;

typedef struct {
    uint64_t semantic_id; /* stable Provider operation identity */
} PyJitRecognition;

typedef struct {
    const PyJitValueRef *value;
    uint8_t escapes;
    uint8_t materialized;
} PyJitRegionValue;

/* Provider-private assumption.  CPython treats kind/key/payload as opaque,
 * uses dependency identities to place the guard, canonicalizes equivalent
 * assumptions by kind/dependencies/key, and unions semantic_property_mask
 * when several projected properties share that canonical assumption. */
typedef struct {
    uint32_t kind;
    const PyJitValueRef *const *dependencies;
    uint8_t dependency_count;
    uint64_t semantic_property_mask;
    const void *canonical_key;
    size_t canonical_key_size;
    const void *payload;
    size_t payload_size;
} PyJitAssumption;

typedef enum {
    PYJIT_PREPARE_ACCEPT = 1,
    PYJIT_PREPARE_UNSUPPORTED_NODE = 2,
    PYJIT_PREPARE_UNSUPPORTED_CONTRACT = 3,
    PYJIT_PREPARE_UNPROFITABLE = 4,
    PYJIT_PREPARE_ERROR = 5,
} PyJitPrepareDecision;

/* C_P plus the assumptions required by that specialization.  On ACCEPT the
 * provider transfers contract ownership to CPython until lowering completes. */
typedef struct {
    uint32_t struct_size;
    PyJitPrepareDecision decision;
    void *contract;
    size_t contract_size;
    const PyJitAssumption *assumptions;
    size_t assumption_count;
} PyJitPrepareResult;

/* Host-scheduled, deduplicated assumptions.  Providers may lower them but
 * cannot choose their placement or side-exit target. */
typedef struct {
    const PyJitRegionView *region;
    const void *contract;
    size_t contract_size;
    const PyJitAssumption *assumptions;
    size_t assumption_count;
} PyJitSpecializationView;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *name;
    const PyJitProviderKey *keys;
    size_t key_count;

    /* Recognition names one operation; projection returns S_H only. */
    PyJitResult (*recognize)(const PyJitRecognizeRequest *request,
                             PyJitRecognition *recognition);
    PyJitResult (*project_semantics)(const PyJitRecognizeRequest *request,
                                     const PyJitRecognition *recognition,
                                     PyJitSemanticSummary *summary);

    /* Invoked only after CPython has constructed HostRegionProof (Pi_H). */
    PyJitResult (*prepare_region)(const PyJitRegionView *region,
                                  PyJitPrepareResult *result);
    PyJitResult (*lower_guards)(const PyJitSpecializationView *specialization,
                                PyJitGuardPlan **plan);
    PyJitResult (*lower_region)(const PyJitSpecializationView *specialization,
                                PyJitExecPlan **plan);

    PyJitResult (*run_guards)(PyJitGuardPlan *plan,
                              PyObject *const *inputs,
                              Py_ssize_t input_count);
    PyJitResult (*run_region)(PyJitExecPlan *plan,
                              PyObject *const *inputs,
                              Py_ssize_t input_count,
                              PyObject **result);

    void (*destroy_contract)(void *contract);
    void (*destroy_guard_plan)(PyJitGuardPlan *plan);
    void (*destroy_exec_plan)(PyJitExecPlan *plan);

    /* Optional, size-gated v3 tail. Execute the supplied Host-scheduled guard
     * batch and region in one callback. Guard failure must precede execution
     * and leave result NULL. guard_status reports the batch result separately
     * from execution (e.g. a replay-safe execution MISS after guards pass).
     * This does not move guard placement or authorize any visible effect.
     * Providers with a shorter struct retain the separate callbacks above. */
    PyJitResult (*run_guarded_region)(PyJitGuardPlan *guards,
                                     PyJitExecPlan *plan,
                                     PyObject *const *inputs,
                                     Py_ssize_t input_count,
                                     PyJitResult *guard_status,
                                     PyObject **result);
    /* Optional generic execution capability. This does not grant fusion,
     * effect or escape legality; Host still independently proves a region.
     * Absent/zero retains the historical multi-node discovery envelope. */
    uint8_t supports_single_node_regions;
} PyJitProviderSpec;

PyAPI_FUNC(int) PyJit_RegisterProvider(const PyJitProviderSpec *spec);
PyAPI_FUNC(int) PyJit_UnregisterProvider(const PyJitProviderSpec *spec);

PyAPI_FUNC(PyObject *) PyJit_ValueGetObservedObject(
    const PyJitValueRef *value);
PyAPI_FUNC(PyTypeObject *) PyJit_ValueGetExactType(
    const PyJitValueRef *value);
PyAPI_FUNC(uint32_t) PyJit_ValueGetId(const PyJitValueRef *value);

PyAPI_FUNC(const PyJitValueRef *) PyJit_NodeGetOperand(
    const PyJitNodeRef *node, uint8_t index);
PyAPI_FUNC(const PyJitValueRef *) PyJit_NodeGetResult(
    const PyJitNodeRef *node, uint8_t index);
PyAPI_FUNC(PyObject *) PyJit_NodeGetKeywordNames(const PyJitNodeRef *node);
PyAPI_FUNC(uint8_t) PyJit_NodeGetPositionalCount(const PyJitNodeRef *node);
/* 0: ordinary callable; 1: exact method descriptor plus receiver. */
PyAPI_FUNC(uint8_t) PyJit_NodeGetCallBindingKind(const PyJitNodeRef *node);
PyAPI_FUNC(uint8_t) PyJit_NodeGetOperandCount(const PyJitNodeRef *node);
PyAPI_FUNC(uint8_t) PyJit_NodeGetResultCount(const PyJitNodeRef *node);
PyAPI_FUNC(uint64_t) PyJit_NodeGetSemanticId(const PyJitNodeRef *node);
PyAPI_FUNC(const PyJitSemanticSummary *) PyJit_NodeGetSemanticSummary(
    const PyJitNodeRef *node);

PyAPI_FUNC(uint8_t) PyJit_RegionGetNodeCount(
    const PyJitRegionView *region);
PyAPI_FUNC(const PyJitNodeRef *) PyJit_RegionGetNode(
    const PyJitRegionView *region, uint8_t index);
PyAPI_FUNC(uint8_t) PyJit_RegionGetInputCount(
    const PyJitRegionView *region);
PyAPI_FUNC(const PyJitRegionValue *) PyJit_RegionGetInput(
    const PyJitRegionView *region, uint8_t index);
PyAPI_FUNC(uint8_t) PyJit_RegionGetResultCount(
    const PyJitRegionView *region);
PyAPI_FUNC(const PyJitRegionValue *) PyJit_RegionGetResult(
    const PyJitRegionView *region, uint8_t index);
PyAPI_FUNC(uint64_t) PyJit_RegionGetStructuralFingerprint(
    const PyJitRegionView *region);

#endif /* !Py_LIMITED_API */

#ifdef __cplusplus
}
#endif
#endif /* !Py_CPYTHON_JIT_H */
