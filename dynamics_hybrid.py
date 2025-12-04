import casadi as ca

# -------------------------
# Parameters and state
# -------------------------
m     = ca.MX.sym('m')
M     = ca.MX.sym('M')
l     = ca.MX.sym('l')
k     = ca.MX.sym('k')
kappa = ca.MX.sym('kappa')
Jpar  = ca.MX.sym('J')     # not used but kept for completeness
g     = ca.MX.sym('g')
p     = ca.vertcat(m, M, l, k, kappa, Jpar, g)

# Generalized coordinates (y–z plane, mean/relative angles from +z)
rCMy  = ca.MX.sym('rCMy')
rCMz  = ca.MX.sym('rCMz')
qmean = ca.MX.sym('qmean')
qrel  = ca.MX.sym('qrel')
q     = ca.vertcat(rCMy, rCMz, qmean, qrel)

# Velocities / accelerations
vCMy  = ca.MX.sym('vCMy')
vCMz  = ca.MX.sym('vCMz')
dqmean= ca.MX.sym('dqmean')
dqrel = ca.MX.sym('dqrel')
dq    = ca.vertcat(vCMy, vCMz, dqmean, dqrel)

aCMy  = ca.MX.sym('aCMy')
aCMz  = ca.MX.sym('aCMz')
ddqmean = ca.MX.sym('ddqmean')
ddqrel  = ca.MX.sym('ddqrel')
ddq   = ca.vertcat(aCMy, aCMz, ddqmean, ddqrel)

# Full state and control
x = ca.vertcat(rCMy, rCMz, qmean, qrel, vCMy, vCMz, dqmean, dqrel)
u = ca.MX.sym('u')
S = ca.vertcat(0, 0, 0, 1)

# Contact flags provided as inputs (previous step)
c1 = ca.MX.sym('c1')
c2 = ca.MX.sym('c2')

# -------------------------
# Lagrangian: T - V
# -------------------------
a = 2*m*l**2/(M+2*m) * ((M+m) - m*ca.cos(qrel))
b = m*l**2/(2*m+4*m) * ((M+m) + m*ca.cos(qrel))
T = 0.5*(M+2*m)*(vCMy**2 + vCMz**2) + 0.5*a*dqmean**2 + 0.5*b*dqrel**2
V = (M+2*m)*g*rCMz + 0.5*k*qrel**2 + kappa*(1 - ca.cos(qrel))
L = T - V

# -------------------------
# Forward kinematics
# -------------------------
phi1 = qmean - 0.5*qrel
phi2 = qmean + 0.5*qrel
rBy = rCMy + 2*m*l/(M+2*m)*ca.sin(qmean)*ca.cos(qrel/2)
rBz = rCMz - 2*m*l/(M+2*m)*ca.cos(qmean)*ca.cos(qrel/2)
r1y = rBy - l*ca.sin(phi1)
r1z = rBz + l*ca.cos(phi1)
r2y = rBy - l*ca.sin(phi2)
r2z = rBz + l*ca.cos(phi2)
r1 = ca.vertcat(r1y, r1z)
r2 = ca.vertcat(r2y, r2z)

# Contact Jacobians wrt q
J1 = ca.jacobian(r1, q)
J2 = ca.jacobian(r2, q)
v1 = ca.mtimes(J1, dq)
v2 = ca.mtimes(J2, dq)

# -------------------------
# Euler–Lagrange dynamics
# -------------------------
grad_q_L  = ca.gradient(L, q)
grad_dq_L = ca.gradient(L, dq)
d_dt_grad_dq_L = ca.jtimes(grad_dq_L, q, dq) + ca.jtimes(grad_dq_L, dq, ddq)
eq = d_dt_grad_dq_L - grad_q_L - S*u
Mmat = ca.jacobian(eq, ddq)
rhs  = -ca.substitute(eq, ddq, 0)
ddq_free = ca.solve(Mmat, rhs, 'symbolicqr')
dxdt_free = ca.vertcat(dq, ddq_free)

# Tip acceleration helper terms
J1dot_qdot = ca.jtimes(v1, q, dq)
J2dot_qdot = ca.jtimes(v2, q, dq)

a1z_free = (J1[1,:] @ ddq_free) + J1dot_qdot[1]
a2z_free = (J2[1,:] @ ddq_free) + J2dot_qdot[1]

# -------------------------
# KKT for single-leg contacts
# -------------------------
def KKT_one(J):
    return ca.vertcat(ca.horzcat(Mmat, -J.T),
                      ca.horzcat(J, ca.MX.zeros(2,2)))

K1 = KKT_one(J1)
b1 = ca.vertcat(rhs, -J1dot_qdot)
sol1 = ca.solve(K1, b1, 'symbolicqr')
ddq_1 = sol1[:4]; lam_1 = sol1[4:]

K2 = KKT_one(J2)
b2 = ca.vertcat(rhs, -J2dot_qdot)
sol2 = ca.solve(K2, b2, 'symbolicqr')
ddq_2 = sol2[:4]; lam_2 = sol2[4:]

# -------------------------
# Contact detection and logic
# -------------------------
eps_g, eps_lam, eps_sep = 1e-7, 1e-9, 1e-9

# --- gate invalid initial contact flags ---
near1 = ca.if_else(r1[1] <= eps_g, 1, 0)
near2 = ca.if_else(r2[1] <= eps_g, 1, 0)
c1s = ca.if_else(c1 > 0.5, near1, 0)
c2s = ca.if_else(c2 > 0.5, near2, 0)

# --- touchdown detection ---
hit1 = ca.if_else(ca.logic_and(r1[1] <= 0, v1[1] < 0), 1, 0)
hit2 = ca.if_else(ca.logic_and(r2[1] <= 0, v2[1] < 0), 1, 0)
both = hit1 * hit2
hit1 = ca.if_else(both, ca.if_else(r1[1] < r2[1], 1, 0), hit1)
hit2 = ca.if_else(both, ca.if_else(r2[1] < r1[1], 1, 0), hit2)

# --- keep active if compressive or pushing up ---
keep1 = ca.if_else(ca.logic_or(lam_1[1] > eps_lam,ca.logic_and(a1z_free < -eps_sep, r1[1] <= eps_g)),1, 0)
keep2 = ca.if_else(ca.logic_or(lam_2[1] > eps_lam,ca.logic_and(a2z_free < -eps_sep, r2[1] <= eps_g)),1, 0)

# --- flight flag (no active contacts) ---
in_flight = ca.if_else(ca.logic_and(c1s < 0.5, c2s < 0.5), 1, 0)
n1_flight = hit1
n2_flight = ca.if_else(hit1 > 0.5, 0, hit2)

n1 = ca.if_else(in_flight > 0.5, n1_flight,
                ca.if_else(c1s > 0.5, keep1, 0))
n2 = ca.if_else(in_flight > 0.5, n2_flight,
                ca.if_else(c2s > 0.5, keep2, 0))

# exclusivity
n1 = ca.if_else(n2 > 0.5, 0, n1)
n2 = ca.if_else(n1 > 0.5, 0, n2)

# -------------------------
# Impact (velocity jump) when prev 0→1
# -------------------------
td1 = ca.if_else(ca.logic_and(c1s < 0.5, n1 > 0.5), 1, 0)
td2 = ca.if_else(ca.logic_and(c2s < 0.5, n2 > 0.5), 1, 0)

rhs_i = ca.vertcat(Mmat @ dq, ca.MX.zeros(2,1))
dq1p = ca.solve(K1, rhs_i, 'symbolicqr')[:4]
dq2p = ca.solve(K2, rhs_i, 'symbolicqr')[:4]
dq_plus = ca.if_else(td1 > 0.5, dq1p,
                     ca.if_else(td2 > 0.5, dq2p, dq))

# -------------------------
# Select stance dynamics (exclusive)
# -------------------------
use1 = ca.if_else(n1 > 0.5, 1, 0)
use2 = ca.if_else(n2 > 0.5, 1, 0)
ddq_stance = ca.if_else(use1 > 0.5, ddq_1,
                        ca.if_else(use2 > 0.5, ddq_2, ddq_free))

lam_out = ca.vertcat(
    ca.if_else(use1 > 0.5, lam_1[0], 0),
    ca.if_else(use1 > 0.5, lam_1[1], 0),
    ca.if_else(use2 > 0.5, lam_2[0], 0),
    ca.if_else(use2 > 0.5, lam_2[1], 0),
)

# build dxdt: positions use post-impact velocity
dxdt = ca.vertcat(dq_plus, ddq_stance)

# -------------------------
# Wrap as CasADi function
# -------------------------
f = ca.Function(
    'fdyn_total',
    [x, u, p, c1, c2],
    [dxdt, lam_out, n1, n2, dq_plus],
    ['x','u','p','c1','c2'],
    ['dxdt','lambda','c1_new','c2_new','dq_plus']
)
