import pennylane as qml
import pennylane.numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp

class SU2EquivarientAnsatz():
    def __init__(self, D:int=2, B:int=1, rep:int=2, active_atoms:int=2):
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

        self.active_atoms = active_atoms  # Number of active atoms
        # Here we only have two active atoms since we fixed the oxygen (which becomes non-active) at the origin
        self.num_qubits = active_atoms * rep
        self.dev = qml.device("default.qubit.jax", wires=self.num_qubits)

    def singlet(self, wires):
        # Encode a 2-qubit rotation-invariant initial state, i.e., the singlet state.
        qml.Hadamard(wires=wires[0])
        qml.PauliZ(wires=wires[0])
        if len(wires) == 2:
            qml.PauliX(wires=wires[1])
            qml.CNOT(wires=wires)

    def equivariant_encoding(self, alpha, data, wires):
        #print("encoding_layer:", wires)
        # data (jax array): cartesian coordinates of atom i
        # alpha (jax array): trainable scaling parameter
        hamiltonian = jnp.einsum("i,ijk", data, self.sigmas)  # Heisenberg Hamiltonian
        
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
            self.singlet(wires=range(self.active_atoms*i, self.active_atoms*(i+1)))

        # Initial encoding
        for i in range(self.num_qubits):
            print(i, input.shape, alphas.shape)
            self.equivariant_encoding(
                #alphas[i, 0], jnp.asarray(input)[i % self.active_atoms, ...], wires=[i]
                alphas[i, 0], jnp.asarray(input), wires=[i]
            )

        # Reuploading model
        for d in range(self.D):
            qml.Barrier()

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
                    alphas[i, d + 1], jnp.asarray(input), wires = [i]
                )
        return qml.expval(self.Observable)

    def compute_func(self, inputs, params, dev=None, with_input_diff=False):
        if dev is None:
            circuit = qml.QNode(self.ansatz, self.dev, interface='jax')
        else:
            assert dev.num_wires == self.dev.num_wires
            circuit = qml.QNode(self.ansatz, dev)

        #print(qml.draw(circuit)(inputs[0], params))
        print("compute_func:", inputs.shape)
        vec_circuit = jax.vmap(circuit, (0, None), 0)
        outputs = vec_circuit(inputs, params)
        diffs = None

        if with_input_diff:
            # diffcircuit = qml.gradients.param_shift(circuit)
            # print(qml.draw(diffcircuit)(inputs[0], params))
            # vec_diffcircuit = jax.vmap(diffcircuit, (0, None), 0)
            # diffs = vec_diffcircuit(inputs, params)

            ## Automatic differentiation
            difffunc = jax.grad(circuit, argnums=0)
            vec_difffunc = jax.vmap(difffunc, (0, None), 0)
            diffs = vec_difffunc(inputs, params)

        return outputs, diffs


if __name__ == '__main__':
    np.random.seed(42)

    D = 2  # Depth of the model
    B = 1  # Number of repetitions inside a trainable layer
    rep = 3  # Number of repeated vertical encoding

    active_atoms = 2  # Number of active atoms
    # Here we only have two active atoms since we fixed the oxygen (which becomes non-active) at the origin
    num_qubits = active_atoms * rep

    ## Initialize trainable params
    weights = np.zeros((num_qubits, D, B))
    weights[0] = np.random.uniform(0, np.pi, 1)
    weights = jnp.array(weights)

    # Encoding weights
    alphas = jnp.array(np.ones((num_qubits, D + 1)))

    # Symmetry-breaking (SB)
    epsilon = jnp.array(np.random.normal(0, 0.001, size=(D, num_qubits)))
    print(weights.shape, alphas.shape, epsilon.shape)
    net_params = {"params": {"weights": weights, "alphas": alphas, "epsilon": epsilon}}

    ## Input data prep
    data = np.array([[[0.101, 0.001, 0.001], [0.001,0.001,0.001]]])
    model = SU2EquivarientAnsatz(D,B,rep,active_atoms)
    output, diff = model.compute_func(data, net_params, with_input_diff=True)

    print(data.shape)
    print(output.shape)
    print(diff.shape)

    print(output)
    print(diff)


