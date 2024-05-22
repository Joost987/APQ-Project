import pennylane as qml
import pennylane.numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from typing import Callable
from jax.example_libraries import optimizers
from functools import partial
from jax import jit


# func = lambda x, f, df: f - np.einsum('...i,...i->...',x,df)
# bdry_conditions = [(x,f), (x,f)]

class DESolver():
    def __init__(self, ansatz, func:Callable, bdry_conditions:list, lr:float=0.01, bdry_coeff:float=1.):
        self.ansatz = ansatz
        self.func = func
        self.order = func.__code__.co_argcount - 2 ## Order of PDE

        self.bdry_inputs, self.bdry_values = zip(*bdry_conditions)
        self.bdry_inputs = np.array(self.bdry_inputs)
        self.bdry_values = np.array(self.bdry_values)
        self.bdry_coeff = bdry_coeff

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)

    # Mean-squared-error loss functions for differential equation & boundary conditions
    @partial(jit, static_argnums=(0,))
    def differential_loss(self, inputs, outputs, diffs):
        print(diffs.shape)
        de_errors = self.func(inputs, outputs, diffs)
        return 0.5 * jnp.mean(de_errors ** 2)
    @partial(jit, static_argnums=(0,))
    def boundary_loss(self, bdry_predictions):
        bdry_errors = bdry_predictions - self.bdry_values
        return 0.5 * jnp.mean(bdry_errors ** 2)

    @partial(jit, static_argnums=(0,))
    def cost(self, weights, inputs, lbda:float=1.):
        outputs, diffs = self.ansatz.compute_func(inputs, weights, with_input_diff=True)
        bdry_predictions, _ = self.ansatz.compute_func(self.bdry_inputs, weights, with_input_diff=False)

        de_loss = self.differential_loss(inputs, outputs, diffs)
        bdry_loss = self.boundary_loss(bdry_predictions)
        return de_loss + lbda * bdry_loss

    # Perform one training step
    @partial(jit, static_argnums=(0,))
    def train_step(self, step_i, opt_state, inputs):
        #print('HII',step_i+1)
        net_params = self.get_params(opt_state)
        loss, grads = jax.value_and_grad(self.cost, argnums=0)(net_params, inputs, lbda=self.bdry_coeff)

        return loss, self.opt_update(step_i, grads, opt_state)


    # Return prediction and loss at inference times, e.g. for testing
    @partial(jit, static_argnums=(0,))
    def inference(self, inputs, opt_state):
        net_params = self.get_params(opt_state)
        f_preds, _ = self.ansatz.compute_func(inputs, net_params, with_input_diff=False)
        l = self.cost(net_params, inputs, lbda=self.bdry_coeff)

        return f_preds, l


    def solve(self, inputs, init_params, test_inputs=None, num_batches=20, batch_size=5):
        if init_params:
            opt_state = self.opt_init(init_params)
        train_record = []
        test_record = []

        for ibatch in range(num_batches):
            # select a batch of training points
            batch = np.random.choice(range(inputs.shape[0]), batch_size, replace=False)

            # preparing the data
            batch_inputs = inputs[batch, ...]

            # perform one training step
            print("num_batches:",num_batches)
            print("opt_state:", opt_state)
            print("batch_inputs shape:", batch_inputs.shape)
            loss, opt_state = self.train_step(num_batches, opt_state, batch_inputs)
            train_record.append(float(loss))

            # computing the test loss and energy predictions
            if test_inputs:
                f_pred, test_loss = self.inference(test_inputs, opt_state)
                test_record.append(float(test_loss))