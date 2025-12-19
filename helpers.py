from integrate import make_stepper,linearize_discrete
from dynamics_hybrid import f

def LQR_mats(x_d,u_d,p,c,dt,N,fdyn=f):
    # get Jacobians as an expression
    F = make_stepper(f, dt)
    A_func, B_func = linearize_discrete(F)

def eigfreqs(A_ctrl):
    print(0)

def construct_initial():
    print(0)