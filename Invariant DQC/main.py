from ansatz import SU2EquivarientAnsatz
from DESolver import DESolver
import numpy as np
from jax import numpy as jnp
from itertools import product
from jax.example_libraries import optimizers
import jax
jax.config.update("jax_enable_x64", True)
import pickle

def prob1(D=3, B=2, rep=2, active_agents=2, with_sb=False, train_size=1024, lr=0.001, bdry_coeff=0.25):
    """
    :param D: Depth of the model
    :param B: Number of repetitions inside a trainable layer
    :param rep: Number of repeated vertical encoding
    :param active_agents: Each agent holds the same (x,y,z) coordinate. All agents enjoy SU(2) invariance
    :return:
    """

    func = lambda x, f, df: 3*f + np.sum(df / x, axis=-1)
    true_f = lambda x: np.exp(-np.sum(x**2)/2.)

    ansatz = SU2EquivarientAnsatz(D, B, rep, active_agents)
    net_params = ansatz.initialize_params(with_sb=with_sb)

    r2 = np.random.uniform(low=0, high=1, size=train_size)
    theta = np.random.uniform(low=0, high=2 * np.pi, size=train_size)
    phi = np.random.uniform(low=0, high=np.pi, size=train_size)

    x = np.sqrt(r2) * np.cos(phi) * np.cos(theta)
    y = np.sqrt(r2) * np.cos(phi) * np.sin(theta)
    z = np.sqrt(r2) * np.sin(phi)

    inputs = np.column_stack([x[:-16],y[:-16],z[:-16]])
    test_inputs = np.column_stack([x[-16:],y[-16:],z[-16:]])

    # Spherical boundary
    bdry_inputs = np.random.uniform(-1,1,size=(1,3))
    bdry_inputs = bdry_inputs / np.linalg.norm(bdry_inputs, axis=1)[:,None]
    bdry_inputs = np.concatenate([bdry_inputs, np.zeros(shape=(1,3))], axis=0)
    bdry_values = np.array([true_f(input) for input in bdry_inputs])
    bdry_conditions = zip(bdry_inputs, bdry_values)

    solver = DESolver(ansatz, func, bdry_conditions, lr=lr, bdry_coeff=bdry_coeff)

    return solver, inputs, test_inputs, net_params

def prob2(D=1, B=1, rep=1, active_agents=2, with_sb=False, train_size=1024, lr=0.001, bdry_coeff=0.25):

    radii = lambda x: jnp.sqrt(jnp.linalg.norm(x, axis=-1))
    ext_func = lambda x, f, df, r: r * jnp.sum(df / x, axis=-1) + 8*f*(0.1 + jnp.tan(8*r))
    func = lambda x, f, df: ext_func(x,f,df,radii(x))

    ext_true_f = lambda x,r: jnp.exp(-0.8*r)*jnp.cos(8*r)
    true_f = lambda x: ext_true_f(x,radii(x)) - 1

    ansatz = SU2EquivarientAnsatz(D, B, rep, active_agents)
    net_params = ansatz.initialize_params(with_sb=with_sb)

    r2 = np.random.uniform(low=0, high=1, size=train_size)
    theta = np.random.uniform(low=0, high=2 * np.pi, size=train_size)
    phi = np.random.uniform(low=0, high=np.pi, size=train_size)

    x = np.sqrt(r2) * np.cos(phi) * np.cos(theta)
    y = np.sqrt(r2) * np.cos(phi) * np.sin(theta)
    z = np.sqrt(r2) * np.sin(phi)

    inputs = np.column_stack([x[:-16],y[:-16],z[:-16]])
    test_inputs = np.column_stack([x[-16:],y[-16:],z[-16:]])

    bdry_inputs = np.random.uniform(-1,1,size=(1,3))
    bdry_inputs = bdry_inputs / np.linalg.norm(bdry_inputs, axis=1)[:,None]
    bdry_inputs = np.concatenate([bdry_inputs, np.zeros(shape=(1,3))], axis=0)
    bdry_values = np.array([true_f(input) for input in bdry_inputs])
    bdry_conditions = zip(bdry_inputs, bdry_values)

    solver = DESolver(ansatz, func, bdry_conditions, lr=lr, bdry_coeff=bdry_coeff)

    return solver, inputs, test_inputs, net_params


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    D = 1
    B = 1
    rep = 1
    filename = f'prob1_D{D}_B{B}_R{rep}.pkl'
    solver, inputs, test_inputs, net_params = prob1(D,B,rep, with_sb=False)

    print("Initial Parameters")
    print(net_params)

    num_batches = 1000
    batch_size = 256
    opt_state = solver.opt_init(net_params)
    train_record = []
    test_record = []

    for ibatch in range(num_batches):
        # select a batch of training points
        batch = np.random.choice(range(inputs.shape[0]), batch_size, replace=False)

        # preparing the data
        batch_inputs = inputs[batch, ...]

        # perform one training step
        loss, opt_state = solver.train_step(num_batches, opt_state, batch_inputs)
        train_record.append(float(loss))

        # computing the test loss and energy predictions
        if test_inputs is not None:
            f_pred, test_loss = solver.inference(test_inputs, opt_state)
            test_record.append(float(test_loss))

        if ibatch % 20 == 0:
            print(f"Batch {ibatch}, train_loss = {loss}", f"test_loss = {test_loss}")


    net_params = solver.get_params(opt_state)
    #print(net_params)

    print("Final Parameters")
    print(net_params)

    with open(filename, 'wb+') as f:  # open a text file
        pickle.dump(net_params, f) # serialize the list

