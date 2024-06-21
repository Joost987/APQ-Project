# library imports
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from qadence import *
from torch import (nn,optim,tensor,ones,zeros,zeros_like,ones_like,sin,cos,exp,rand,linspace,manual_seed,atan2)
from torch.autograd import grad
# random seed
manual_seed(42)
#Solving Laplace equation with u(0,y)=sin(pi*y),u(x,0)=0,u(X,y)=0,u(x,Y)=0 for x,y in [0,1]
# helper function to calculate derivatives

def calc_derivative(outputs, inputs) -> tensor:
    """
    Returns the derivative of a function output
    with respect to its inputs
    """
    if not inputs.requires_grad:
        inputs.requires_grad = True
    return grad(
        inputs=inputs,
        outputs=outputs,
        grad_outputs=ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        )[0]

class DomainSampling(nn.Module):
    """
    Collocation points sampling from domains uses uniform random sampling.
    Problem-specific MSE loss function for solving the 2D Laplace equation.
    """
    def __init__(self, net: nn.Module | QNN, n_inputs: int = 2, n_colpoints: int = 20):
        super().__init__()
        self.net = net
        self.n_colpoints = n_colpoints
        self.n_inputs = n_inputs

    def left_boundary(self) -> tensor: # u(0,y)=0
        sample = rand(size=(self.n_colpoints, self.n_inputs))
        sample[:, 0] = 0.0
        return self.net(sample).pow(2).mean()
    def right_boundary(self) -> tensor: # u(L,y)=0
        sample = rand(size=(self.n_colpoints, self.n_inputs))
        sample[:, 0] = 1.0
        return self.net(sample).pow(2).mean()
    def top_boundary(self) -> tensor: # u(x,H)=0
        sample = rand(size=(self.n_colpoints, self.n_inputs))
        sample[:, 1] = 1.0
        return self.net(sample).pow(2).mean()
    
    def bottom_boundary(self) -> tensor: # u(x,0)=f(x)
        sample = rand(size=(self.n_colpoints, self.n_inputs))
        sample[:, 1] = 0.0
        return (self.net(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()
    

    

    
    def interior(self) -> tensor: # uxx+uyy=0
        sample = rand(size=(self.n_colpoints, self.n_inputs), requires_grad=True)
        first_both = calc_derivative(self.net(sample), sample)
        second_both = calc_derivative(first_both, sample)
        return (second_both[:, 0] + second_both[:, 1]).pow(2).mean()
    
    #def interior_(self)
    


LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_POINTS = 150
# define a simple DQC model
ansatz = hea(n_qubits=N_QUBITS, depth=DEPTH)
# parallel Fourier feature map
split = N_QUBITS // len(VARIABLES)
fm = kron(
    *[
        feature_map(n_qubits=split, support=support, param=param)
            for param, support in zip(
                VARIABLES,
                [
                    list(list(range(N_QUBITS))[i : i + split])
                    for i in range(N_QUBITS)
                if i % split == 0
                ],
            )
            ]
            )
# choosing a cost function
obs = ising_hamiltonian(n_qubits=N_QUBITS)

# building the circuit and the quantum model
circuit = QuantumCircuit(N_QUBITS, chain(fm, ansatz))
model = QNN(circuit=circuit, observable=obs, inputs=VARIABLES)
# using Adam as an optimiser of choice
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# get the collocation sampling for loss calculation
sol = DomainSampling(net=model, n_inputs=2, n_colpoints=100)
# training
lossarr=np.zeros(1000)
for epoch in range(1000):
    opt.zero_grad()
    loss = (
    sol.left_boundary()
    + sol.right_boundary()
    + sol.top_boundary()
    + sol.bottom_boundary()
    + sol.interior()
    )
    loss.backward()
    opt.step()
    lossarr[epoch]=loss.item()
# visualisation and comparison of results
single_domain = linspace(0, 1, steps=N_POINTS)
domain = tensor(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
(exp(-np.pi * domain[:, 0]) * sin(np.pi * domain[:, 1]))
.reshape(N_POINTS, N_POINTS)
.T
)
# DQC solution
dqc_sol = model(domain).reshape(N_POINTS, N_POINTS).detach().numpy()
#classical solver solution
from pde import CartesianGrid, solve_laplace_equation
grid=CartesianGrid([[0, 1]] * 2, N_POINTS,periodic=[False,False])
bcs = [({"value": "sin(pi*y)" }, {"value": 0}),({"value": 0},{"value": 0})]
res = solve_laplace_equation(grid, bcs)
# plot results
fig, ax = plt.subplots(1, 3, figsize=(7, 7))
ax[0].imshow(analytic_sol, cmap="turbo")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Analytical solution u(x,y)")
ax[1].imshow(dqc_sol, cmap="turbo")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("DQC solution u(x,y)")
ax[2].imshow(res.data.transpose(), cmap="turbo")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].set_title("Classical  u(x,y)")
plt.show()
plt.plot(lossarr)
plt.xlabel("Iterations")
plt.ylabel("Log loss")
plt.yscale("log")
plt.savefig("PDELoss.pdf")
plt.show()
#res.plot()
