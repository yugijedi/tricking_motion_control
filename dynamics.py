import casadi as ca

#parameters
m = ca.MX.sym('m')
M = ca.MX.sym('M')
l = ca.MX.sym('l')
k = ca.MX.sym('k')
kappa = ca.MX.sym('kappa')
J = ca.MX.sym('J')
p = ca.vertcat(m,M,l,k,kappa,J)

#states and controls
qmean = ca.MX.sym('qmean')
qrel = ca.MX.sym('qrel')
dqmean = ca.MX.sym('dqmean')
dqrel = ca.MX.sym('dqrel')
ddqmean = ca.MX.sym('ddqmean')
ddqrel = ca.MX.sym('ddqrel')
q = ca.vertcat(qmean,qrel)
dq = ca.vertcat(dqmean,dqrel)
ddq = ca.vertcat(ddqmean,ddqrel)
x = ca.vertcat(qmean,qrel,dqmean,dqrel)
u = ca.MX.sym('u')

#assemble dynamics function and save it
a = 2*m*l**2/(M+2*m) * ((M+m) - m*ca.cos(qrel))
b = m*l**2/(2*m+4*m) * ((M+m) + m*ca.cos(qrel))
T = 1/2*a*dqmean**2 + 1/2*b*dqrel**2
V = 1/2*k*qrel**2 + kappa*(1-ca.cos(qrel))
L = T - V
grad_q_L = ca.gradient(L,q)
grad_dq_L = ca.gradient(L,dq)
d_dt_grad_dq_L = ca.jtimes(grad_dq_L, q, dq) + ca.jtimes(grad_dq_L, dq, ddq)
eq = d_dt_grad_dq_L - grad_q_L - ca.vertcat(0,u) #Euler-Lagrange equations, equals zero
ddq_sol = -ca.solve(ca.jacobian(eq,ddq),ca.substitute(eq,ddq,0),'symbolicqr')
#print(ddq_sol)
#print(ca.n_nodes(ddq_sol))
ddq_sol = ca.cse(ddq_sol)
#print(ca.n_nodes(ddq_sol))
dxdt = ca.vertcat(dq,ddq_sol)
f = ca.Function('f',
                [x, u, p], [dxdt],
                ['x', 'u', 'p'], ['dxdt'])