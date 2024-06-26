import pennylane as qml
import pennylane.numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from functools import partial

class SU2EquivarientAnsatz():
    def __init__(self, D:int=2, B:int=1, rep:int=2, active_agents:int=2):
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1.0j], [1.0j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        self.sigmas = jnp.array(np.array([X, Y, Z]))  # Vector of Pauli matrices
        self.sigmas_sigmas = jnp.array(
            np.array(
                [np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)]  # Vector of tensor products of Pauli matrices
            )
        )

        # Invariant observable
        self.Heisenberg = [
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
        ]
        self.Observable = qml.Hamiltonian(np.ones((3)), self.Heisenberg)

        ############ Setup ##############
        self.D = D  # Depth of the model
        self.B = B  # Number of repetitions inside a trainable layer
        self.rep = rep  # Number of repeated vertical encoding

        self.active_agents = active_agents  # Number of active atoms
        # Here we only have two active atoms since we fixed the oxygen (which becomes non-active) at the origin
        self.num_qubits = active_agents * rep
        self.dev = qml.device("default.qubit.jax", wires=self.num_qubits)

    def initialize_params(self, with_sb=False):
        #weights = np.array(np.random.normal(0, 0.01, size=(self.num_qubits, self.D, self.B)))
        weights = np.zeros((self.num_qubits, self.D, self.B))
        weights[0] = np.random.rand()
        weights = jnp.array(weights)

        # Encoding weights
        alphas = jnp.array(np.ones((self.num_qubits, self.D + 1)))

        # Symmetry-breaking (SB)
        if with_sb:
            epsilon = jnp.array(np.random.normal(0, 0.001, size=(self.D, self.num_qubits)))
        else:
            epsilon = None
        epsilon = jax.lax.stop_gradient(epsilon)

        coeffs = jnp.array(np.random.normal(size=(2)))

        net_params = {"params": {"weights": weights, "alphas": alphas, "epsilon": epsilon, "coeffs": coeffs}}
        return net_params

    def singlet(self, wires):
        # Encode a 2-qubit rotation-invariant initial state, i.e., the singlet state.
        assert len(wires) == 2, "For now we only support initial singlet state (2 active agents)"
        qml.Hadamard(wires=wires[0])
        qml.PauliX(wires=wires[1])
        qml.CNOT(wires=wires)
        qml.PauliZ(wires=wires[0])

    def equivariant_encoding(self, alpha, data, wires):
        #print("encoding_layer:", wires)
        # data (jax array): cartesian coordinates of atom i
        # alpha (jax array): trainable scaling parameter
        hamiltonian = jnp.einsum('i,ijk->jk', data, self.sigmas)  # Heisenberg Hamiltonian

        
        U = jax.scipy.linalg.expm(-1.0j * alpha * hamiltonian / 2)
        qml.QubitUnitary(U, wires=wires, id="E")

    def trainable_layer(self, weight, wires):
        #print("trainable_layer:", wires)
        hamiltonian = jnp.einsum("ijk->jk", self.sigmas_sigmas)
        U = jax.scipy.linalg.expm(-1.0j * weight * hamiltonian)
        qml.QubitUnitary(U, wires=wires, id="U")

    def noise_layer(self, epsilon, wires):
        #print("noise_layer:", wires)
        for _, w in enumerate(wires):
            qml.RZ(epsilon[_], wires=[w])

    #@qml.qnode(self.dev, interface="jax")
    def ansatz(self, input, params):
        weights = params["params"]["weights"]
        alphas = params["params"]["alphas"]
        epsilon = params["params"]["epsilon"]


        # Initial state
        for i in range(self.rep):
            #print(i, self.active_atoms, [self.active_atoms * i, self.active_atoms * i + 1])
            #self.singlet(wires=[self.active_atoms * i, self.active_atoms*(i+1)])
            self.singlet(wires=range(self.active_agents * i, self.active_agents * (i + 1)))

        # Initial encoding
        for i in range(self.num_qubits):
            self.equivariant_encoding(
                #alphas[i, 0], jnp.asarray(input)[i % self.active_atoms, ...], wires=[i]
                alphas[i, 0], jnp.asarray(input)[:], wires=[i]
            )

        # Reuploading model
        for d in range(self.D):
            #qml.Barrier()

            for b in range(self.B):
                # Even layer
                for i in range(0, self.num_qubits - 1, 2):
                    self.trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % self.num_qubits])

                # Odd layer
                for i in range(1, self.num_qubits, 2):
                    self.trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % self.num_qubits])

            # Symmetry-breaking
            if epsilon is not None:
                self.noise_layer(epsilon[d, :], range(self.num_qubits))

            # Encoding
            for i in range(self.num_qubits):
                self.equivariant_encoding(
                    #alphas[i, d + 1], jnp.asarray(input)[i % self.active_atoms, ...], wires=[i]
                    alphas[i, d + 1], jnp.asarray(input)[...], wires = [i]
                )
        return qml.expval(self.Observable)

    def compute_func(self, inputs, params, dev=None, with_input_diff=False, with_input_hessian=False):
        print("Compute_func module inputs", inputs.shape)

        if dev is None:
            circuit = qml.QNode(self.ansatz, self.dev, interface='jax')
        else:
            assert dev.num_wires == self.dev.num_wires
            circuit = qml.QNode(self.ansatz, dev)

        coeffs = params["params"]["coeffs"]
        vec_circuit = jax.vmap(circuit, (0, None), 0)
        outputs = coeffs[0]*vec_circuit(inputs, params) + coeffs[1]
        diffs = None
        hessians = None

        if with_input_diff:
            # Automatic differentiation
            difffunc = jax.grad(circuit, argnums=0)
            vec_difffunc = jax.vmap(difffunc, (0, None), 0)
            diffs = coeffs[0]*vec_difffunc(inputs, params)

        if with_input_hessian:
            ## Second order
            hessianfunc = jax.hessian(circuit, argnums=0)
            vec_hessianfunc = jax.vmap(hessianfunc, (0, None), 0)
            hessians = coeffs[0] * jax.numpy.diagonal(vec_hessianfunc(inputs, params),  axis1=-2, axis2=-1)#.sum(axis=1)

        return outputs, diffs, hessians


