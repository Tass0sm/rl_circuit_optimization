import numpy as np
import gymnasium as gym

from enum import Enum
from ray.rllib.env.env_context import EnvContext
from itertools import repeat
from gymnasium.spaces import Discrete, Graph, GraphInstance

from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.random import random_circuit
from qiskit.compiler.transpiler import transpile
from qiskit.transpiler.passmanager import PassManager
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.transpiler.passes.optimization import (
    Optimize1qGates,
    Optimize1qGatesDecomposition,
    Collect2qBlocks,
    CollectMultiQBlocks,
    ConsolidateBlocks,
    CommutationAnalysis,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    CXCancellation,
    Optimize1qGatesSimpleCommutation,
    OptimizeSwapBeforeMeasure,
    RemoveResetInZeroState,
    RemoveDiagonalGatesBeforeMeasure,
    CrosstalkAdaptiveSchedule,
    HoareOptimizer,
    TemplateOptimization,
    InverseCancellation,
    Collect1qRuns,
    EchoRZXWeylDecomposition,
    CollectLinearFunctions,
    ResetAfterMeasureSimplification,
    OptimizeCliffords,
    CollectCliffords
)

BASIS_UNARY_GATES=["u", "u1", "u2", "u3", "clifford", "linear_function", "id", "measure"]
BASIS_BINARY_GATES=["cx"]

TRANSPILER_PASSES=[
    Optimize1qGates(),
    Optimize1qGatesDecomposition(),
    Collect2qBlocks(),
    CollectMultiQBlocks(),
    ConsolidateBlocks(),
    CommutationAnalysis(),
    CommutativeCancellation(),
    CommutativeInverseCancellation(),
    CXCancellation(),
    Optimize1qGatesSimpleCommutation(),
    OptimizeSwapBeforeMeasure(),
    RemoveResetInZeroState(),
    RemoveDiagonalGatesBeforeMeasure(),
    # Require's backend information:
    # CrosstalkAdaptiveSchedule(),
    HoareOptimizer(),
    TemplateOptimization(),
    # InverseCancellation(),
    Collect1qRuns(),
    # EchoRZXWeylDecomposition(),
    CollectLinearFunctions(),
    ResetAfterMeasureSimplification(),
    OptimizeCliffords(),
    CollectCliffords()
]

class QuantumCircuitEnv(gym.Env):
    """RL environment for quantum circuits.

    Observation Space:
    Graph of Circuit
    1. Nodes are the operations
    2. Edges are wires labels

    Still no parameters are included in the gate representation.

    Actions: Circuit Transformations

    """
    def __init__(self, config: EnvContext):


        # Z-Rotation, Phased-X and Controlled-Not (CNOT) gates.
        # gate_space = Discrete(3)
        # gate_space = Box(low=-100, high=100, shape=(3,))
        self._width = 2
        self._depth = 2

        self.num_unary_basis_gates = len(BASIS_UNARY_GATES)
        self.num_binary_basis_gates = len(BASIS_BINARY_GATES)

        # TODO: Make this all more readable. It is too terse.
        # an identifier for any item in the graph
        gate_space = Discrete(self.num_unary_basis_gates + 2 * self.num_binary_basis_gates)
        # a binary label for to distinguish wires coming into a two-qubit gate
        edge_space = Discrete(self._width)
        self._unary_gate_type_to_id = dict(zip(BASIS_UNARY_GATES, range(self.num_unary_basis_gates)))
        self._binary_gate_type_to_id = dict(zip(BASIS_BINARY_GATES, range(self.num_unary_basis_gates, self.num_unary_basis_gates + 2 * self.num_binary_basis_gates)))
        self.observation_space = Graph(node_space=gate_space, edge_space=edge_space)

        num_transformations = len(TRANSPILER_PASSES)
        self._action_type_to_transpiler_pass = dict(zip(range(num_transformations), TRANSPILER_PASSES))
        self.action_space = Discrete(num_transformations)

    def _encode_circuit_dag_node(self, node):
        n = node.op.name

        if len(node.qargs) == 1:
            return self._unary_gate_type_to_id[n]
        elif len(node.qargs) == 2:
            arg1 = node.qargs[0]
            arg2 = node.qargs[1]
            idx1 = self._circuit.find_bit(arg1)
            idx2 = self._circuit.find_bit(arg2)

            if idx1 < idx2:
                return self._binary_gate_type_to_id[n]
            else:
                return self._binary_gate_type_to_id[n] + self.num_binary_basis_gates
        else:
            return None

    def _encode_circuit_dag(self, circuit_dag):
        op_nodes = circuit_dag.op_nodes()
        op_node_ids = [self._encode_circuit_dag_node(node) for node in op_nodes]

        is_between_ops = lambda a: isinstance(a[0], DAGOpNode) and isinstance(a[1], DAGOpNode)
        edges = list(filter(is_between_ops, circuit_dag.edges(nodes=op_nodes)))
        index_edges = [[op_nodes.index(a), op_nodes.index(b)] for a, b, _ in edges]

        def get_edge_wire(e):
            s1 = set(e[0].qargs)
            s2 = set(e[1].qargs)
            qubit = s1.intersection(s2).pop()
            return self._circuit.find_bit(qubit).index

        edge_wires = list(map(get_edge_wire, edges))

        return GraphInstance(nodes=np.array(op_node_ids),
                             edges=np.array(edge_wires),
                             edge_links=np.array(index_edges))

    def _get_info(self):
        # TODO: add more
        return {
            "dag_size": self._circuit_dag.size(),
            "depth": self._circuit.depth()
        }

    def reset(self, *, seed=None, options=None):
        """
        Starts a new episode by randomly generating a new circuit. Returns the
        first agent observation, which is the circuit dag encoded as a member of
        the observation space. Also returns metrics (TODO)
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        c = random_circuit(2, 2, measure=True)
        c = transpile(c, basis_gates=BASIS_GATES, optimization_level=0)

        self._circuit = c
        self._circuit_dag = circuit_to_dag(self._circuit)

        o = self._encode_circuit_dag(self._circuit_dag)
        i = self._get_info()

        return o, i

    def _compute_reward(self, action, state, new_state):
        # TODO
        return -(new_state.depth() - state.depth())

    def step(self, action):
        p = self._action_type_to_transpiler_pass[action]
        pm = PassManager(p)

        new_circuit = pm.run(self._circuit)
        new_circuit_dag = circuit_to_dag(new_circuit)

        o = self._encode_circuit_dag(new_circuit_dag)
        r = self._compute_reward(action, self._circuit_dag, new_circuit_dag)
        terminated = True
        truncated = False

        self._circuit = new_circuit
        self._circuit_dag = new_circuit_dag
        i = self._get_info()
        return o, r, terminated, truncated, i
