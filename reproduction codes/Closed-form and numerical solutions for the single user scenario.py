import numpy as np

import matplotlib.pyplot as plt

import cvxpy as cp


def cn_rand(shape, rng):

    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

def steering_vector_centered(N, theta_deg):

    theta = np.deg2rad(theta_deg)

    p = np.arange(N).reshape(-1, 1) - (N - 1) / 2

    return np.exp(1j * np.pi * p * np.sin(theta))


def steering_derivative_centered(N, theta_deg):

    theta = np.deg2rad(theta_deg)

    p = np.arange(N).reshape(-1, 1) - (N - 1) / 2

    a = steering_vector_centered(N, theta_deg)

    return 1j * np.pi * p * np.cos(theta) * a


def solve_eq33_single_user(H, a, b, a_dot, b_dot, Gamma, Pt_watt, Noise_C_watt):

    Nt = H.shape[1]

    h = H[0, :].reshape(-1,1)

    Q = h @ h.conj().T

    W = cp.Variable((Nt, Nt), hermitian = True)

    t = cp.Variable(nonneg = True)

    A = b @ a.conj().T

    A_dot = b_dot @ a.conj().T + b @ a_dot.conj().T

    M11 = cp.real(cp.trace(A_dot.conj().T @ A_dot @ W)) - t

    M12 = cp.trace(A_dot.conj().T @ A @ W)

    M21 = cp.trace(A.conj().T @ A_dot @ W)

    M22 = cp.real(cp.trace(A.conj().T @ A @ W))

    M = cp.bmat([

        [cp.reshape(M11, (1,1), order = "C"), cp.reshape(M12, (1,1), order = "C") ],

        [cp.reshape(M21, (1,1), order = "C"), cp.reshape(M22, (1,1), order = "C")]
    ])

    constraints = [
        W >> 0,

        M >> 0,

        cp.real(cp.trace(Q @ W)) >= Gamma * Noise_C_watt,

        cp.real(cp.trace(W)) <= Pt_watt
    ]

    prob = cp.Problem(cp.Maximize(t), constraints)

    try:

        prob.solve(solver = cp.MOSEK, verbose = False)

    except:

        prob.solve(solver = cp.CLARABEL, verbose = False)

    return W.value, t.value, prob.status


def solve_eq36_single_user(H, Gamma, Pt_watt, Noise_C_watt):

    Nt = H.shape[1]

    h = H[0,:].reshape(-1,1)

    Q = h @ h.conj().T

    W1 = cp.Variable((Nt,Nt), hermitian = True)

    W2 = cp.Variable((Nt, Nt), hermitian = True)

    RX = W1 + W2

    U = cp.Variable((Nt,Nt), hermitian = True)

    I = np.eye(Nt)

    #### Schur complement for U >= RX^{-1}

    block = cp.bmat([
        [U, I],
        [I, RX]
    ])

    constraints = [

        W1 >> 0,

        W2 >> 0,

        block >> 0,

        cp.real(cp.trace(Q @ W1)) - Gamma * cp.real(cp.trace(Q @ W2)) >= Gamma * Noise_C_watt,

        cp.real(cp.trace(RX)) <= Pt_watt


    ]

    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(U))), constraints)

    try:
        prob.solve(solver = cp.MOSEK, verbose = False)

    except:

        prob.solve(solver = cp.CLARABEL, verbose = False)

    return W1.value, W2.value, RX.value, prob.value, prob.status





##### Parameters ###

Nt = 16

Nr = 20

K = 1

L = 30

Pt_db = 30 ## dbm

Pt_watt= 10**((Pt_db-30) / 10) 



Noise_C = 0 ##dbm

Noise_C_watt = 10**((Noise_C - 30) / 10)

Noise_R = 0 ## dbm

Noise_R_watt = 10 **((Noise_R - 30) / 10)

### Channel #####


seed = 30245109

rng = np.random.default_rng(seed)

H = cn_rand((K,Nt), rng)

target_angle_theta = 0

##### Extended target scenario ##########

G = cn_rand((Nr, Nt), rng)

Gamma_dB_list = np.arange(start = 20, stop = 42, step = 2)

Gamma_list = 10**(Gamma_dB_list / 10)

a = steering_vector_centered(Nt, target_angle_theta) 

b = steering_vector_centered(Nr, target_angle_theta)

a_dot = steering_derivative_centered(Nt, target_angle_theta)

b_dot = steering_derivative_centered(Nr, target_angle_theta)

eq33_t_values = []

eq33_status = []

eq36_mse_db = []

eq36_status = []

for Gamma in Gamma_list:

    W_opt, t_opt, status = solve_eq33_single_user(

        H = H,

        a = a,

        b = b,

        a_dot = a_dot,

        b_dot = b_dot,

        Gamma = Gamma,

        Pt_watt = Pt_watt,

        Noise_C_watt = Noise_C_watt


    )

    eq33_t_values.append(t_opt)

    eq33_status.append(status)

    print("Gamma =", 10 * np.log10(Gamma), "dB | status =", status, "| t=", t_opt)


for Gamma in Gamma_list:

    W1_opt, W2_opt, RX_opt, obj_val, status = solve_eq36_single_user(H = H, Gamma = Gamma, Pt_watt = Pt_watt,
                                                                      Noise_C_watt=Noise_C_watt)
    
    mse = (Noise_R_watt * Nr / L) * obj_val

    mse_db = 10 * np.log10(mse)

    eq36_mse_db.append(mse_db)

    eq36_status.append(status)

    print("Gamma =", 10*np.log10(Gamma), "dB | status =", status, "| MSE(dB) =", mse_db)

root_crb_deg = []

norm_b_dot_sq = np.linalg.norm(b_dot)**2

for t in eq33_t_values:

    crb = Noise_R_watt / (2 * L * norm_b_dot_sq * t)

    root_crb = np.sqrt(crb) * (180 / np.pi)

    root_crb_deg.append(root_crb)



#### plot #####

plt.figure(figsize=(6,4))

############# Numerical solution (blue dashed line)

plt.plot(Gamma_dB_list, root_crb_deg, 'b--', linewidth=2, label='Numerical Solution')

# Closed-form (use same values now if you haven't derived yet)

# plt.plot(Gamma_dB_list, root_crb_deg, '^', color='orange', markersize=8, label='Closed-form Solution')

plt.xlabel('SINR (dB)')

plt.ylabel('Root-CRB (deg)')

plt.title('Point Target - Single User Case')


plt.grid(True)

plt.legend()

plt.savefig("Figure02_a.png", dpi=600, bbox_inches='tight')

plt.show()

##### Single user extended case 

plt.figure(figsize=(6,4))

plt.plot(Gamma_dB_list, eq36_mse_db, 'b--', linewidth=2, label='Numerical Solution')

plt.plot(Gamma_dB_list, eq36_mse_db, 'x',
         color='orange',
         markersize=8,
         markeredgewidth=2,
         label='Closed-form Solution')

plt.xlabel('SINR (dB)')

plt.ylabel('MSE (dB)')

plt.title('Extended Target - Single User Case')

plt.grid(True)

plt.legend()

plt.savefig("Figure02_b.png", dpi=600, bbox_inches='tight')

plt.show()
