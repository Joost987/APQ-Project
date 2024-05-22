from ansatz import SU2EquivarientAnsatz
from DESolver import DESolver
import numpy as np
from jax import numpy as jnp
from itertools import product
from jax.example_libraries import optimizers
import jax
jax.config.update("jax_enable_x64", True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #func = lambda x, f, df: 3*f + np.sum(df, axis=-1)[:,0]
    func = lambda x, f, df: 3*f + 0.5*np.sum(df / x, axis=-1)
    true_f = lambda x: np.exp(-np.sum(x**2))

    D = 2  # Depth of the model
    B = 1  # Number of repetitions inside a trainable layer
    rep = 2  # Number of repeated vertical encoding

    active_atoms = 2  # Number of active atoms. Since we only have one coordinate (x,y,z) now, we suppose all "atoms" share the same coordinate. The point is to establish SU(2) invariance
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
    net_params = {"params": {"weights": weights, "alphas": alphas, "epsilon": epsilon}}

    ## Input data prep
    # inputs = np.random.uniform(0,1,size=(1000,1,3))
    # test_inputs = np.random.uniform(0,1,size=(16,1,3))
    # inputs = np.repeat(inputs, active_atoms, axis=1)
    # test_inputs = np.repeat(test_inputs, active_atoms, axis=1)

    inputs = np.random.uniform(0,1,size=(1000,3))
    test_inputs = np.random.uniform(0,1,size=(16,3))

    # Square boundary
    # bdry_inputs = np.array(list(product([0,1], repeat=3)))
    # bdry_values = np.array([true_f(input) for input in bdry_inputs])

    # Spherical boundary
    bdry_inputs = np.random.uniform(-1,1,size=(1,3))
    bdry_inputs = bdry_inputs / np.linalg.norm(bdry_inputs, axis=1)
    bdry_values = np.array([true_f(input) for input in bdry_inputs])

    #bdry_inputs = np.repeat(bdry_inputs.reshape(8,3), active_atoms, axis=1)
    bdry_conditions = zip(bdry_inputs, bdry_values)

    print(inputs.shape, test_inputs.shape, bdry_inputs.shape, bdry_values.shape)

    ansatz = SU2EquivarientAnsatz(D,B,rep,active_atoms)
    solver = DESolver(ansatz, func, bdry_conditions)

    #solver.solve(inputs, net_params, test_inputs, num_batches=20, batch_size=5)
    num_batches = 1
    batch_size = 5
    opt_init, opt_update, get_params = optimizers.adam(0.01)
    opt_state = opt_init(net_params)
    train_record = []
    test_record = []

    for ibatch in range(num_batches):
        # select a batch of training points
        batch = np.random.choice(range(inputs.shape[0]), batch_size, replace=False)

        # preparing the data
        batch_inputs = inputs[batch, ...]

        # perform one training step
        print("num_batches:", num_batches)
        #print("opt_state:", opt_state)
        print("batch_inputs shape:", batch_inputs.shape)
        loss, opt_state = solver.train_step(num_batches, opt_state, batch_inputs)
        train_record.append(float(loss))

        # computing the test loss and energy predictions
        if test_inputs is not None:
            print('test_inputs shape', test_inputs.shape)
            f_pred, test_loss = solver.inference(test_inputs, opt_state)
            test_record.append(float(test_loss))


    print(train_record)
    print(test_record)

    net_params = self.get_params(opt_state)
