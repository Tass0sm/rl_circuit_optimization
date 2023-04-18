import numpy as np
import gymnasium as gym

from enum import Enum
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Discrete, Graph, GraphInstance

from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.random import random_circuit
from qiskit.compiler.transpiler import transpile
from qiskit.transpiler.passmanager import PassManager
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.transpiler.passes.optimization.commutative_cancellation import CommutativeCancellation

BASIS_GATES=["u", "cx", "id", "measure"]
TRANSPILER_PASSES=[
    CommutativeCancellation()
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
        self._depth = 2

        num_basis_gates = len(BASIS_GATES)
        # an identifier for any item in the graph
        gate_space = Discrete(num_basis_gates)
        # a binary label for to distinguish wires coming into a two-qubit gate
        edge_space = Discrete(self._depth)
        self._gate_type_to_name = dict(zip(BASIS_GATES, range(num_basis_gates)))
        self.observation_space = Graph(node_space=gate_space, edge_space=edge_space)

        num_transformations = len(TRANSPILER_PASSES)
        self._action_type_to_transpiler_pass = dict(zip(range(num_transformations), TRANSPILER_PASSES))
        self.action_space = Discrete(num_transformations)

    def _encode_circuit_dag(self, circuit_dag):
        op_nodes = circuit_dag.op_nodes()
        op_node_type_names = [node.op.name for node in op_nodes]
        op_node_types = [self._gate_type_to_name[n] for n in op_node_type_names]

        is_between_ops = lambda a: isinstance(a[0], DAGOpNode) and isinstance(a[1], DAGOpNode)
        edges = list(filter(is_between_ops, circuit_dag.edges(nodes=op_nodes)))
        index_edges = [[op_nodes.index(a), op_nodes.index(b)] for a, b, _ in edges]

        def get_edge_wire(e):
            s1 = set(e[0].qargs)
            s2 = set(e[1].qargs)
            qubit = s1.intersection(s2).pop()
            return self._circuit.find_bit(qubit).index

        edge_wires = list(map(get_edge_wire, edges))

        return GraphInstance(nodes=np.array(op_node_types),
                             edges=np.array(edge_wires),
                             edge_links=np.array(index_edges))

    def _get_info(self):
        # TODO
        return {}

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
        i = self._get_info()
        terminated = False

        self._circuit = new_circuit
        self._circuit_dag = new_circuit_dag
        return o, r, terminated, False, i
