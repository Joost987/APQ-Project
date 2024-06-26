from ansatz import SU2EquivarientAnsatz
from DESolver import DESolver
import numpy as np
from jax import numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)
import pickle
import matplotlib.pyplot as plt

def prob1(D=2, B=1, rep=2, active_agents=2):
    func = lambda x, f, df: 3*f + 0.5*np.sum(df / x, axis=-1)
    true_f = lambda x: np.exp(-np.sum(x**2))

    rs = np.linspace(0,1, num=51)
    inputs = []
    for r in rs:
        theta = np.random.uniform(low=0, high=2 * np.pi, size=16)
        phi = np.random.uniform(low=0, high=np.pi, size=16)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        inputs.append(np.column_stack([x,y,z]))
    inputs = np.array(inputs)

    ansatz = SU2EquivarientAnsatz(D,B,rep,active_agents)

    return ansatz, rs, inputs, true_f

def prob2(D=2, B=1, rep=2, active_agents=2):

    radii = lambda x: jnp.sqrt(jnp.linalg.norm(x, axis=-1))
    ext_func = lambda x, f, df, r: r * jnp.sum(df / x, axis=-1) + 8*f*(0.1 + jnp.tan(8*r))
    func = lambda x, f, df: ext_func(x,f,df,radii(x))

    ext_true_f = lambda x,r: jnp.exp(-0.8*r)*jnp.cos(8*r)
    true_f = lambda x: ext_true_f(x,radii(x)) - 1

    rs = np.linspace(0,1, num=51)
    inputs = []
    for r in rs:
        theta = np.random.uniform(low=0, high=2 * np.pi, size=16)
        phi = np.random.uniform(low=0, high=np.pi, size=16)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        inputs.append(np.column_stack([x,y,z]))
    inputs = np.array(inputs)

    ansatz = SU2EquivarientAnsatz(D,B,rep,active_agents)

    return ansatz, rs, inputs, true_f

if __name__ == '__main__':
    D = 1
    B = 1
    rep = 1
    filename = f'prob1_D{D}_B{B}_R{rep}.pkl'
    with open(filename, 'rb') as f:
        net_params = pickle.load(f)
    active_agents = 2 # Fixed
    #ansatz = SU2EquivarientAnsatz(D,B,rep,active_agents)
    ansatz, rs, inputs, true_f = prob1(D, B, rep, active_agents)

    outputs_by_rad = []
    outputs_se_by_rad = []
    true_outputs_by_rad = []
    for i,r in enumerate(rs):
        sym_inputs = inputs[i]

        sym_outputs,_,_ = ansatz.compute_func(sym_inputs, net_params)
        true_sym_outputs = [true_f(inp) for inp in sym_inputs]


        outputs_by_rad.append(np.mean(sym_outputs))
        outputs_se_by_rad.append(np.std(sym_outputs)/np.sqrt(len(sym_outputs)))
        true_outputs_by_rad.append(np.mean(true_sym_outputs))
        assert np.std(true_sym_outputs) < 1e-5

    outputs_by_rad = np.array(outputs_by_rad)
    plt.plot(rs, true_outputs_by_rad, label="Analytical Solution")
    plt.errorbar(rs, outputs_by_rad, yerr=outputs_se_by_rad, label="Inv-DQC Solution")
    plt.xlim((0,1))
    plt.legend()
    plt.show()