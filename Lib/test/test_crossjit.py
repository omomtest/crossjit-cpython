"""Host-semantic CrossJIT tests using a domain-neutral synthetic Provider."""

import os
import sys
import types
import unittest

from test import support
from test.support.bytecode_helper import AssemblerTestCase

import _testinternalcapi as internal


class Box:
    add_calls = 0

    def __init__(self, value, token=1, *, miss=False, error=False):
        self.value = value
        self.token = token
        self.crossjit_miss = miss
        self.crossjit_error = error

    def __add__(self, other):
        type(self).add_calls += 1
        return Box(self.value + other.value, self.token)


class ExplodingToken:
    def __eq__(self, other):
        raise RuntimeError("synthetic guard error")


def new_fused(filename):
    namespace = {}
    exec(compile(
        "def fused(a, b, c, count):\n"
        "    for _ in range(count):\n"
        "        result = (a + b) + c\n"
        "    return result\n",
        filename,
        "exec",
    ), namespace)
    return namespace["fused"]


def keep_right(left, right):
    return right


def new_physically_invalid_fused(filename):
    namespace = {"keep_right": keep_right}
    exec(compile(
        "def fused(a, b, c, count):\n"
        "    for _ in range(count):\n"
        # Keeping ``a`` below the arithmetic DAG makes an entry value live at
        # region exit.  Its semantic topology matches new_fused(), but its
        # trace-specific physical mapping must be rejected.
        "        result = keep_right(a, (a + b) + c)\n"
        "    return result\n",
        filename,
        "exec",
    ), namespace)
    return namespace["fused"]


def new_long_fused(filename, length=4):
    namespace = {}
    names = tuple(f"v{i}" for i in range(length + 1))
    expression = names[0]
    for name in names[1:]:
        expression = f"({expression} + {name})"
    exec(compile(
        f"def fused({', '.join(names)}, count):\n"
        "    for _ in range(count):\n"
        f"        result = {expression}\n"
        "    return result\n",
        filename,
        "exec",
    ), namespace)
    return namespace["fused"]


@support.requires_jit_enabled
@unittest.skipUnless(
    os.environ.get("PYTHON_JIT_CROSSJIT") == "1",
    "requires PYTHON_JIT_CROSSJIT=1",
)
class CrossJitSyntheticProviderTests(unittest.TestCase):
    def setUp(self):
        internal.crossjit_synthetic_register(Box)
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        Box.add_calls = 0
        self.functions = []

    def tearDown(self):
        for function in self.functions:
            internal.invalidate_executors(function.__code__)
        internal.clear_executor_deletion_list()
        internal.crossjit_synthetic_unregister()

    def run_new(self, filename, *, semantic=0, contract=1, decision=1,
                assumption=0, lowering=0, max_nodes=0, token=1, combined=0):
        internal.crossjit_synthetic_configure(
            semantic, contract, decision, assumption, lowering, max_nodes, combined)
        function = new_fused(filename)
        self.functions.append(function)
        result = function(
            Box(1, token), Box(2, token), Box(3, token),
            internal.TIER2_THRESHOLD * 2)
        self.assertEqual(result.value, 6)
        return function

    def run_long(self, filename, *, decision=1, max_nodes=0, length=4):
        internal.crossjit_synthetic_configure(
            0, 1, decision, 0, 0, max_nodes)
        function = new_long_fused(filename, length)
        self.functions.append(function)
        result = function(
            *(Box(i) for i in range(1, length + 2)),
            internal.TIER2_THRESHOLD * 2)
        self.assertEqual(result.value, (length + 1) * (length + 2) // 2)
        return function

    def test_nonclosed_prefix_does_not_mask_admitted_suffix(self):
        internal.crossjit_synthetic_configure(0, 1, 4, 0, 0, 2)
        namespace = {}
        exec(compile('def f(a,b,c,d,count):\n    for _ in range(count):\n        r=(a+b)+(c+d)\n    return r\n',
                     'closed-suffix.py', 'exec'), namespace)
        function = namespace['f']
        self.functions.append(function)
        result = function(Box(1), Box(2), Box(3), Box(4), internal.TIER2_THRESHOLD*3)
        self.assertEqual(result.value, 10)
        stats = internal.crossjit_stats()
        self.assertEqual(stats['rewritten_regions'], 1)
        self.assertGreater(stats['exec_hits'], 0)
        self.assertEqual(stats['exec_misses'], 0)

    def test_long_generic_dag_guard_replay_and_ownership(self):
        for length in (5, 8, 12):
            with self.subTest(length=length):
                internal.crossjit_reset_stats()
                function = self.run_long(f"long-{length}.py", length=length)
                stats = internal.crossjit_stats()
                self.assertEqual(stats["recognized_nodes"], length)
                self.assertEqual(stats["rewritten_regions"], 1)
                self.assertGreater(stats["exec_hits"], 0)
                self.assertEqual(stats["provider_first_recognized"], length)
                values = tuple(Box(i, token=9) for i in range(1, length + 2))
                Box.add_calls = 0
                result = function(*values, 2)
                self.assertEqual(result.value, (length + 1) * (length + 2) // 2)
                self.assertEqual(Box.add_calls, length * 2)
                self.assertGreater(internal.crossjit_stats()["guard_misses"], 0)

    def test_late_constant_live_in_ownership_gc_and_replay(self):
        import gc
        import _opcode
        import weakref

        for nodes in (2, 5, 8):
            with self.subTest(nodes=nodes):
                internal.crossjit_reset_stats()
                internal.crossjit_synthetic_configure(0, 1, 1)
                namespace = {}
                expression = "(a + b)" + " + 42.0" * (nodes - 1)
                exec("def fused(a, b, count):\n"
                     "    for _ in range(count):\n"
                     f"        result = {expression}\n"
                     "    return result\n", namespace)
                function = namespace["fused"]
                literal = Box(3)
                function.__code__ = function.__code__.replace(
                    co_consts=tuple(literal if value == 42 else value
                                    for value in function.__code__.co_consts))
                self.functions.append(function)
                literal_refs = sys.getrefcount(literal)
                values = (Box(1), Box(2))
                input_refs = tuple(map(sys.getrefcount, values))
                result = function(*values, internal.TIER2_THRESHOLD * 2)
                self.assertEqual(result.value, 3 * nodes)
                stats = internal.crossjit_stats()
                self.assertEqual(stats["rewritten_regions"], 1)
                self.assertGreater(stats["exec_hits"], 0)
                self.assertEqual(tuple(map(sys.getrefcount, values)), input_refs)
                self.assertGreater(sys.getrefcount(literal), literal_refs)
                executors = []
                for offset in range(0, len(function.__code__.co_code), 2):
                    try:
                        executor = _opcode.get_executor(function.__code__, offset)
                    except ValueError:
                        continue
                    if executor is not None:
                        executors.append(executor)
                self.assertTrue(any(literal is obj for executor in executors
                                    for obj in gc.get_referents(executor)))
                # Both guards and ordinary replay see the original literal.
                Box.add_calls = 0
                self.assertEqual(function(Box(1, token=9), values[1], 2).value,
                                 3 * nodes)
                self.assertEqual(Box.add_calls, 2 * nodes)
                self.assertGreater(internal.crossjit_stats()["guard_misses"], 0)
                internal.invalidate_executors(function.__code__)
                del executors, executor
                internal.clear_executor_deletion_list()
                support.gc_collect()
                self.assertEqual(sys.getrefcount(literal), literal_refs)
                # After removing the code owner, a literal/executor cycle
                # must be collectable through the new plan traversal edge.
                function.__code__ = function.__code__.replace()
                function(*values, internal.TIER2_THRESHOLD * 2)
                for offset in range(0, len(function.__code__.co_code), 2):
                    try:
                        executor = _opcode.get_executor(function.__code__, offset)
                    except ValueError:
                        continue
                    if executor is not None and any(
                            obj is literal for obj in gc.get_referents(executor)):
                        literal.owner = executor
                        break
                self.assertTrue(hasattr(literal, "owner"))
                observed = weakref.ref(literal)
                self.functions.remove(function)
                namespace.clear()
                del function, literal, namespace, executor
                support.gc_collect()
                self.assertIsNone(observed())

    def test_optional_single_node_capability_and_legacy_size(self):
        for capability, combined, expected in ((0, 0, False), (1, 0, True),
                                                (1, 1, True), (1, -1, False)):
            with self.subTest(capability=capability, combined=combined):
                namespace = {}
                exec("def fused(a, b, count):\n"
                     "    for _ in range(count):\n"
                     "        result = a + b\n"
                     "    return result\n", namespace)
                function = namespace['fused']
                self.functions.append(function)
                internal.crossjit_reset_stats()
                internal.crossjit_synthetic_configure(0, 1, 1, 0, 0, 0,
                                                      combined, capability)
                self.assertEqual(function(Box(1), Box(2),
                                          internal.TIER2_THRESHOLD * 2).value, 3)
                stats = internal.crossjit_stats()
                self.assertEqual(bool(stats['exec_hits']), expected)
                if expected:
                    Box.add_calls = 0
                    self.assertEqual(function(Box(1, token=9), Box(2), 2).value, 3)
                    self.assertEqual(Box.add_calls, 2)
                    self.assertGreater(internal.crossjit_stats()['guard_misses'], 0)

    def test_region_after_loop_exhaustion_with_live_trace_entry_stack(self):
        """A POP_ITER side trace has an iterator/index at its entry."""
        internal.crossjit_synthetic_configure(0, 1, 1)

        def fused(a, b, c, count):
            for _ in range(count):
                pass
            return (a + b) + c

        self.functions.append(fused)
        values = (Box(1), Box(2), Box(3))
        # Enough backedges per call let the loop/POP_ITER side trace warm
        # before RESUME, including under PYTHON_JIT_STRESS. Keep runtime
        # trigger policy unchanged; zero/short loops are validated below.
        for _ in range(10000):
            result = fused(*values, 10)
        self.assertEqual(result.value, 6)
        stats = internal.crossjit_stats()
        self.assertGreater(stats["physical_proof_validated"], 0)
        self.assertGreater(stats["exec_hits"], 0)
        for count in (0, 1, 10):
            self.assertEqual(fused(*values, count).value, 6)
        Box.add_calls = 0
        result = fused(Box(1, token=9), Box(2), Box(3), 2)
        self.assertEqual(result.value, 6)
        self.assertEqual(Box.add_calls, 2)
        self.assertGreater(internal.crossjit_stats()["guard_misses"], 0)

    def test_temporary_local_escape_stays_materialized(self):
        # COPY/stack motion support does not grant permission to erase a
        # Python-visible STORE_FAST or its surviving local value.
        namespace = {}
        exec("def fused(a, b, c, d, count):\n"
             "    for _ in range(count):\n"
             "        saved = a + b\n"
             "        result = (saved + c) + d\n"
             "    return saved, result\n", namespace)
        function = namespace["fused"]
        self.functions.append(function)
        internal.crossjit_synthetic_configure(0, 1, 1, 0, 0, 0)
        internal.crossjit_reset_stats()
        saved, result = function(Box(1), Box(2), Box(3), Box(4),
                                 internal.TIER2_THRESHOLD * 2)
        self.assertEqual((saved.value, result.value), (3, 10))
        self.assertIsNot(saved, result)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["recognized_nodes"], 3)
        self.assertEqual(stats["rewritten_regions"], 1)
        self.assertGreater(stats["exec_hits"], 0)

    def new_transient_local_dag(self, length, *, movement="clear", seeded=False):
        assembler = AssemblerTestCase()
        insts = [("RESUME", 0)]
        if seeded:
            # A preexisting local's decref cannot be suppressed, even when its
            # result would subsequently be deleted by this expression.
            insts.extend((("LOAD_FAST", 2), ("STORE_FAST", 3)))
        insts.extend((("LOAD_FAST", 0), ("LOAD_FAST", 1), ("BINARY_OP", 0)))
        for _ in range(length - 1):
            insts.append(("STORE_FAST", 3))
            if movement == "clear":
                insts.append(("LOAD_FAST_AND_CLEAR", 3))
            else:
                insts.extend((("LOAD_FAST", 3), ("DELETE_FAST", 3)))
            insts.extend((("LOAD_FAST", 2), ("BINARY_OP", 0)))
        insts.append(("RETURN_VALUE", None))
        metadata = dict(name="transient_dag", qualname="transient_dag",
                        consts={}, names={}, varnames={"a": 0, "b": 1, "c": 2, "tmp": 3},
                        cellvars={}, freevars={}, fasthidden={},
                        argcount=3, posonlyargcount=0, kwonlyargcount=0, firstlineno=1)
        code = assembler.get_code_object(
            f"transient-{length}-{movement}-{seeded}.py",
            assembler.seq_from_insts(insts), metadata)
        function = types.FunctionType(code, {})
        self.functions.append(function)
        return function

    def test_transient_local_forwarding_guard_replay_and_ownership(self):
        for length in (4, 5, 8):
            for movement in ("clear", "delete"):
                with self.subTest(length=length, movement=movement):
                    internal.crossjit_reset_stats()
                    internal.crossjit_synthetic_configure(0, 1, 1)
                    function = self.new_transient_local_dag(length, movement=movement)
                    def exercise(values, count):
                        for _ in range(count):
                            result = function(*values)
                        return result
                    self.functions.append(exercise)
                    values = (Box(1), Box(2), Box(3))
                    result = exercise(values, internal.TIER2_RESUME_THRESHOLD * 2)
                    self.assertEqual(result.value, 3 * length)
                    stats = internal.crossjit_stats()
                    self.assertEqual(stats["recognized_nodes"], length)
                    self.assertEqual(stats["rewritten_regions"], 1)
                    self.assertGreater(stats["exec_hits"], 0)
                    self.assertGreater(stats["transient_local_exec_hits"], 0)
                    refs = tuple(sys.getrefcount(v) for v in values)
                    self.assertEqual(exercise(values, 100).value, 3 * length)
                    self.assertEqual(tuple(sys.getrefcount(v) for v in values), refs)
                    Box.add_calls = 0
                    self.assertEqual(exercise((Box(1, token=9), Box(2), Box(3)), 100).value,
                                     3 * length)
                    self.assertEqual(Box.add_calls, 100 * length)
                    self.assertGreater(internal.crossjit_stats()["guard_misses"],
                                       stats["guard_misses"])

    def test_transient_local_with_preexisting_value_replays(self):
        internal.crossjit_synthetic_configure(0, 1, 1)
        function = self.new_transient_local_dag(4, seeded=True)
        values = (Box(1), Box(2), Box(3))
        refs = tuple(sys.getrefcount(v) for v in values)
        for _ in range(internal.TIER2_RESUME_THRESHOLD * 2):
            self.assertEqual(function(*values).value, 12)
        stats = internal.crossjit_stats()
        self.assertGreater(stats["guard_misses"], 0)
        self.assertGreater(stats["transient_local_guard_misses"], 0)
        # A newly formed suffix may execute after the first store has already
        # decref'ed the old local normally. The full suppressed-store plan
        # must instead reject its nonempty local at entry.
        self.assertEqual(tuple(sys.getrefcount(v) for v in values), refs)

    def test_fallible_operation_with_live_local_stays_materialized(self):
        namespace = {}
        exec("def fused(a, b, c, count):\n"
             "    for _ in range(count):\n"
             "        tmp = a + b\n"
             "        result = tmp + c\n"
             "        del tmp\n"
             "    return result\n", namespace)
        function = namespace["fused"]
        self.functions.append(function)
        internal.crossjit_synthetic_configure(0, 1, 1)
        self.assertEqual(function(Box(1), Box(2), Box(3),
                                  internal.TIER2_THRESHOLD * 2).value, 6)
        self.assertEqual(internal.crossjit_stats()["exec_hits"], 0)

    def test_stack_reference_movement_preserves_dag_identity(self):
        assembler = AssemblerTestCase()
        for movement in ("COPY", "SWAP"):
            with self.subTest(movement=movement):
                insts = [("RESUME", 0), ("LOAD_FAST", 0),
                         ("LOAD_FAST", 1), ("BINARY_OP", 0)]
                for _ in range(7):
                    if movement == "COPY":
                        insts.append(("COPY", 1))
                    else:
                        insts.extend((("LOAD_FAST", 2), ("SWAP", 2)))
                    insts.append(("BINARY_OP", 0))
                insts.append(("RETURN_VALUE", None))
                metadata = dict(name="stack_dag", qualname="stack_dag",
                                consts={}, names={}, varnames={"a": 0, "b": 1, "c": 2},
                                cellvars={}, freevars={}, fasthidden={},
                                argcount=3, posonlyargcount=0, kwonlyargcount=0,
                                firstlineno=1)
                code = assembler.get_code_object(
                    f"stack-{movement}.py", assembler.seq_from_insts(insts), metadata)
                function = types.FunctionType(code, {})
                self.functions.append(function)
                internal.crossjit_synthetic_configure(0, 1, 1, 0, 0, 0)
                internal.crossjit_reset_stats()
                for _ in range(internal.TIER2_RESUME_THRESHOLD * 2):
                    result = function(Box(1), Box(2), Box(3))
                self.assertEqual(result.value, 384 if movement == "COPY" else 24)
                stats = internal.crossjit_stats()
                self.assertEqual(stats["recognized_nodes"], 8)
                self.assertEqual(stats["rewritten_regions"], 1)
                self.assertGreater(stats["exec_hits"], 0)
                before = stats["guard_misses"]
                Box.add_calls = 0
                result = function(Box(1, token=9), Box(2, token=9), Box(3, token=9))
                self.assertEqual(result.value, 384 if movement == "COPY" else 24)
                self.assertEqual(Box.add_calls, 8)
                self.assertGreater(internal.crossjit_stats()["guard_misses"], before)

    def test_semantic_proof_specialization_and_runtime_protocol(self):
        first = self.run_new("synthetic-a.py", contract=101)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["recognized_nodes"], 2)
        self.assertEqual(stats["semantic_class_created"], 1)
        self.assertEqual(stats["rewritten_regions"], 1)
        self.assertEqual(stats["published_regions"], 1)
        self.assertEqual(stats["assumptions_before_dedup"], 2)
        self.assertEqual(stats["assumptions_after_dedup"], 1)
        self.assertGreater(stats["guard_hits"], 0)
        self.assertGreater(stats["exec_hits"], 0)

        # Guard placement is Host-owned and runs before the Provider consumes
        # an input.  A changed private token side-exits to ordinary Python.
        Box.add_calls = 0
        result = first(Box(1, token=9), Box(2, token=9), Box(3, token=9), 2)
        self.assertEqual(result.value, 6)
        self.assertGreater(internal.crossjit_stats()["guard_misses"], 0)
        self.assertEqual(Box.add_calls, 4)

        # Provider execution can soft-miss, or report an error, without
        # violating the generic Host protocol.
        Box.add_calls = 0
        result = first(Box(1, miss=True), Box(2), Box(3), 2)
        self.assertEqual(result.value, 6)
        self.assertGreater(internal.crossjit_stats()["exec_misses"], 0)
        self.assertEqual(Box.add_calls, 4)
        Box.add_calls = 0
        with self.assertRaisesRegex(RuntimeError, "synthetic execution error"):
            first(Box(1, error=True), Box(2), Box(3), 2)
        # One pre-executor loop iteration runs normally; ERROR itself is not
        # replayed as another original-operation iteration.
        self.assertEqual(Box.add_calls, 2)

        # Guard ERROR propagates without consuming inputs or replaying the
        # original operations.
        Box.add_calls = 0
        with self.assertRaisesRegex(RuntimeError, "synthetic guard error"):
            first(Box(1, token=ExplodingToken()), Box(2), Box(3), 2)
        self.assertEqual(Box.add_calls, 2)

        # A different C_P tag with identical S_H reuses Pi_H and rebuilds
        # only Provider specialization.
        self.run_new("synthetic-b.py", contract=202, token=7)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["semantic_class_created"], 1)
        self.assertGreaterEqual(stats["semantic_class_reused"], 1)
        self.assertGreaterEqual(stats["provider_specialization_rebuilt"], 1)

        # The second node now projects a may-alias result.  It remains a legal
        # final result, but changes S_H and therefore invalidates proof reuse.
        self.run_new("synthetic-c.py", semantic=2, contract=303)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["semantic_class_created"], 2)
        self.assertGreaterEqual(stats["semantic_class_miss"], 2)

        # Structured rejection is observable and leaves execution generic.
        self.run_new("synthetic-reject.py", contract=404, decision=3)
        stats = internal.crossjit_stats()
        self.assertGreaterEqual(stats["reject_unsupported_contract"], 1)

        # Contracts are temporary; published guard/exec plans live until the
        # owning executors are invalidated and are then destroyed once.
        self.assertGreaterEqual(
            internal.crossjit_synthetic_stats()["contract_destroyed"], 3)
        for function in self.functions:
            internal.invalidate_executors(function.__code__)
        internal.clear_executor_deletion_list()
        lifetime = internal.crossjit_synthetic_stats()
        self.assertEqual(lifetime["guard_destroyed"], 3)
        self.assertEqual(lifetime["exec_destroyed"], 3)
        self.functions.clear()

    def test_combined_guard_protocol_and_short_spec_compatibility(self):
        for mode in (1, -1):
            with self.subTest(combined=mode):
                internal.crossjit_reset_stats()
                function = self.run_new(f"combined-{mode}.py", combined=mode)
                stats = internal.crossjit_stats()
                self.assertGreater(stats["guard_hits"], 0)
                self.assertEqual(stats["guard_hits"], stats["exec_hits"])
                self.assertEqual(stats["combined_guard_exec_calls"] > 0, mode == 1)
                values = (Box(1), Box(2), Box(3))
                references = tuple(sys.getrefcount(value) for value in values)
                for _ in range(64):
                    self.assertEqual(function(*values, 2).value, 6)
                self.assertEqual(tuple(sys.getrefcount(value) for value in values), references)
                misses = stats["exec_misses"]
                result = function(Box(1, token=9), Box(2), Box(3), 2)
                self.assertEqual(result.value, 6)
                self.assertGreater(internal.crossjit_stats()["guard_misses"], 0)
                self.assertEqual(internal.crossjit_stats()["exec_misses"], misses)
                result = function(Box(1, miss=True), Box(2), Box(3), 2)
                self.assertEqual(result.value, 6)
                self.assertGreater(internal.crossjit_stats()["exec_misses"], misses)
                with self.assertRaisesRegex(RuntimeError, "synthetic execution error"):
                    function(Box(1, error=True), Box(2), Box(3), 2)
                with self.assertRaisesRegex(RuntimeError, "synthetic guard error"):
                    function(Box(1, token=ExplodingToken()), Box(2), Box(3), 2)
                internal.invalidate_executors(function.__code__)
                internal.clear_executor_deletion_list()

    def test_combined_guard_rejects_provider_protocol_violation(self):
        for mode, message in ((2, "execution proceeded after failed guards"),
                              (3, "combined guard returned invalid status")):
            with self.subTest(mode=mode):
                internal.crossjit_synthetic_configure(0, 1, 1, 0, 0, 0, mode)
                function = new_fused(f"invalid-combined-{mode}.py")
                self.functions.append(function)
                with self.assertRaisesRegex(SystemError, message):
                    function(Box(1), Box(2), Box(3), internal.TIER2_THRESHOLD * 2)

    def test_guarded_semantic_property_assumption_shapes(self):
        # One guarded property can depend on exactly one assumption.
        self.run_new("one-property-one-assumption.py", assumption=1)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["assumptions_before_dedup"], 1)
        self.assertEqual(stats["assumptions_after_dedup"], 1)
        synthetic = internal.crossjit_synthetic_stats()
        one_property_mask = synthetic["last_guard_property_mask"]
        self.assertEqual(synthetic["last_guard_assumption_count"], 1)
        self.assertNotEqual(one_property_mask, 0)

        # Several properties may share one canonical assumption.
        before = internal.crossjit_stats()
        self.run_new("properties-share-assumption.py", assumption=3)
        after = internal.crossjit_stats()
        self.assertEqual(
            after["assumptions_before_dedup"] -
            before["assumptions_before_dedup"], 1)
        self.assertEqual(
            after["assumptions_after_dedup"] -
            before["assumptions_after_dedup"], 1)
        shared = internal.crossjit_synthetic_stats()
        shared_property_mask = shared["last_guard_property_mask"]
        self.assertEqual(shared["last_guard_assumption_count"], 1)
        self.assertNotEqual(shared_property_mask, one_property_mask)

        # One property may require independently placed assumptions.
        before = internal.crossjit_stats()
        self.run_new("one-property-two-assumptions.py", assumption=2)
        after = internal.crossjit_stats()
        self.assertEqual(
            after["assumptions_before_dedup"] -
            before["assumptions_before_dedup"], 2)
        self.assertEqual(
            after["assumptions_after_dedup"] -
            before["assumptions_after_dedup"], 2)
        multiple = internal.crossjit_synthetic_stats()
        self.assertEqual(multiple["last_guard_assumption_count"], 2)
        self.assertEqual(multiple["last_guard_property_mask"],
                         one_property_mask)

        # Canonical-equivalent descriptions become one guard and their
        # property dependencies are unioned on the scheduled assumption.
        before = internal.crossjit_stats()
        self.run_new("canonical-assumption-dedup.py", assumption=0)
        after = internal.crossjit_stats()
        self.assertEqual(
            after["assumptions_before_dedup"] -
            before["assumptions_before_dedup"], 2)
        self.assertEqual(
            after["assumptions_after_dedup"] -
            before["assumptions_after_dedup"], 1)
        canonical = internal.crossjit_synthetic_stats()
        self.assertEqual(canonical["last_guard_assumption_count"], 1)
        self.assertEqual(canonical["last_guard_property_mask"],
                         shared_property_mask)

        # Projected properties are unconditional when the Provider declares
        # no assumptions; no guard is emitted for this specialization.
        internal.crossjit_synthetic_reset()
        before = internal.crossjit_stats()
        self.run_new("unconditional-properties.py", assumption=4)
        after = internal.crossjit_stats()
        self.assertEqual(
            after["assumptions_before_dedup"] -
            before["assumptions_before_dedup"], 0)
        self.assertEqual(
            after["assumptions_after_dedup"] -
            before["assumptions_after_dedup"], 0)
        unconditional = internal.crossjit_synthetic_stats()
        self.assertEqual(unconditional["last_guard_assumption_count"], 0)

    def test_effect_boundaries_prefix_authority_and_fault_lifetime(self):
        # Releasing the GIL is a Host-visible interior effect and prevents
        # virtual elimination even when every object property is otherwise
        # fresh, replayable, and virtualizable.
        self.run_new("may-release-gil.py", semantic=3)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["formed_regions"], 0)
        self.assertEqual(stats["rewritten_regions"], 0)
        self.assertEqual(
            internal.crossjit_synthetic_stats()["prepare_called"], 0)

        # UNSUPPORTED_NODE permits Host-owned repeated prefix shortening:
        # four nodes -> closed three-node candidates -> accepted two-node prefix.
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        self.run_long("prefix-retry.py", max_nodes=2)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["reject_unsupported_node"], 3)
        self.assertEqual(stats["rewritten_regions"], 1)
        self.assertEqual(stats["published_regions"], 1)
        self.assertEqual(
            internal.crossjit_synthetic_stats()["prepare_called"], 4)

        # Broader discovery must preserve a separately measured profitable
        # prefix. Provider only rejects the supplied candidate; Host chooses
        # and independently reproves four -> three -> two nodes.
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        self.run_long("profitable-prefix.py", decision=4, max_nodes=2)
        stats = internal.crossjit_stats()
        self.assertEqual(stats["reject_unprofitable"], 3)
        self.assertEqual(stats["rewritten_regions"], 1)
        self.assertGreater(stats["exec_hits"], 0)
        self.assertEqual(
            internal.crossjit_synthetic_stats()["prepare_called"], 4)

        # UNSUPPORTED_CONTRACT does not authorize prefix retries.
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        self.run_long("contract-reject.py", decision=3)
        self.assertEqual(
            internal.crossjit_synthetic_stats()["prepare_called"], 1)
        self.assertEqual(internal.crossjit_stats()["rewritten_regions"], 0)

        # If guard lowering succeeds and execution lowering declines, the
        # unpublished guard and temporary contract are each destroyed once.
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        self.run_new("exec-lowering-miss.py", lowering=1)
        lifetime = internal.crossjit_synthetic_stats()
        self.assertEqual(lifetime["contract_destroyed"], 1)
        self.assertEqual(lifetime["guard_destroyed"], 1)
        self.assertEqual(lifetime["exec_destroyed"], 0)
        self.assertEqual(internal.crossjit_stats()["rewritten_regions"], 0)

        # The entry scheduler rejects a Provider dependency on an internal
        # virtual value: it is unavailable at the Host-selected guard point.
        internal.crossjit_reset_stats()
        internal.crossjit_synthetic_reset()
        internal.crossjit_synthetic_configure(0, 1, 1, 5, 0, 0)
        function = new_fused("internal-guard-dependency.py")
        self.functions.append(function)
        result = function(
            Box(1), Box(2), Box(3), internal.TIER2_THRESHOLD * 2)
        self.assertEqual(result.value, 6)
        self.assertEqual(internal.crossjit_stats()["rewritten_regions"], 0)
        rejected = internal.crossjit_synthetic_stats()
        self.assertGreaterEqual(rejected["prepare_called"], 1)
        self.assertEqual(rejected["contract_destroyed"], 1)
        self.assertEqual(rejected["guard_destroyed"], 0)
        self.assertEqual(rejected["exec_destroyed"], 0)

    def test_semantic_reuse_never_reuses_physical_mapping(self):
        self.run_new("physical-valid.py", assumption=1)
        valid_stats = internal.crossjit_stats()
        self.assertEqual(valid_stats["semantic_class_created"], 1)
        self.assertEqual(valid_stats["physical_proof_attempted"], 1)
        self.assertEqual(valid_stats["physical_proof_validated"], 1)
        self.assertEqual(valid_stats["rewritten_regions"], 1)

        internal.crossjit_synthetic_configure(0, 2, 1, 1)
        function = new_physically_invalid_fused("physical-invalid.py")
        self.functions.append(function)
        result = function(
            Box(1), Box(2), Box(3), internal.TIER2_THRESHOLD * 2)
        self.assertEqual(result.value, 6)

        stats = internal.crossjit_stats()
        self.assertGreaterEqual(stats["semantic_class_reused"], 1)
        self.assertEqual(stats["physical_proof_attempted"], 2)
        self.assertEqual(stats["physical_proof_validated"], 1)
        self.assertEqual(stats["rewritten_regions"], 1)
        self.assertGreaterEqual(stats["rejected_regions"], 1)


if __name__ == "__main__":
    unittest.main()
