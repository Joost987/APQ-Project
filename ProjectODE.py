#ODE: du\dx+lambda u (k+tan(lambda x))=0 u(0)=u_0
#solution : u(x)=exp(-klambda*x)*cos(lambda x)+const
#lambda=8, k=0.1, u_0=1
#equidistant optimization grid of 20 points, starting from x=0 with max time 0.9
#N=6 qubits, cost function C=sum Z_j, Variational circuit:standard hardware efficient ansatz d=5
#Adapative stochastic gradient descent using ADAM, with automatic differentiation enabled by analytical derivatives

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from qadence import *
from torch import nn, optim, tensor, zeros_like, ones_like, linspace, manual_seed, cos
from torch.autograd import grad
from torch import tan, cos, sin,exp

manual_seed(404)
def closure():
    opt.zero_grad()
    loss = loss_fn(inputs=cp, outputs=model(cp))
    loss.backward()
    return loss

lam=20
k=0.1
u0=1
max_time=0.9



N_QUBITS, DEPTH, LEARNING_RATE, N_POINTS = 6, 5, 0.05, 20
#N_QUBITS, DEPTH, LEARNING_RATE, N_POINTS = 6, 16, 0.05, 100
cp=tensor( #equidistant grid
    np.linspace(0,max_time,num=N_POINTS).reshape([N_POINTS,1]), requires_grad=True
    ).float()

sample_points = linspace(0, max_time, steps=100).reshape(-1, 1)

ansatz = hea(n_qubits=N_QUBITS, depth=DEPTH)
fm = feature_map(n_qubits=N_QUBITS,param="x", fm_type = BasisSet.CHEBYSHEV,reupload_scaling = ReuploadScaling.TOWER)
obs=add(Z(i) for i in range(N_QUBITS))
circuit = QuantumCircuit(N_QUBITS, chain(fm, ansatz))
model = QNN(circuit=circuit, observable=obs, inputs=["x"],diff_mode=DiffMode.AD)
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)



def loss_fn(inputs: tensor, outputs: tensor) -> tensor:
    dfdx =grad(inputs=inputs, outputs=outputs.sum(), create_graph=True)[0] 
    fb=u0-outputs[0].item() #floating boundary handling
    outputs+=fb

    ode_loss = dfdx + (lam * outputs * (k+tan(lam*inputs)))
    return ode_loss.pow(2).mean()

#initialise parameters
minloss=1
dqc_sol=None
Nits=2000
lossarr=np.zeros(Nits)
for epoch in range(Nits):
    opt.zero_grad()
    
    loss = loss_fn(inputs=cp, outputs=model(cp))
    if (loss<minloss):
        minloss=loss
        dqc_sol = model(sample_points).detach().numpy()+u0-model(zeros_like(sample_points))[0].item()
    if (epoch % 50 == 0):
        print(epoch, loss.item())

    if (loss.item()<0.01):
        break
        
    loss.backward()
    opt.step()
    lossarr[epoch]=loss.item()




analytic_sol = ( #u(x)=exp(-klambda*x)*cos(lambda x)
    exp(-k*lam*sample_points)*cos(lam*sample_points)

)


try:
    if dqc_sol==None:
        dqc_sol = model(sample_points).detach().numpy()+u0-model(zeros_like(sample_points))[0].item()
except(ValueError):
    pass

x_data = sample_points.detach().numpy()
plt.figure(figsize=(4, 4))
plt.plot(x_data, analytic_sol.flatten(), color="gray", label="Exact solution")
plt.plot(x_data, dqc_sol.flatten(), color="orange", label="DQC solution")

plt.plot()
plt.xlim(0,max_time)
plt.xlabel("x")
plt.ylabel("df | dx")
plt.legend()
#plt.savefig("ODEHighDepth2.pdf")
plt.show()

plt.plot(lossarr)
plt.xlabel("Iterations")
plt.ylabel("Log loss")
plt.yscale("log")
#plt.savefig("ODE2HighDepthLoss.pdf")
plt.show()