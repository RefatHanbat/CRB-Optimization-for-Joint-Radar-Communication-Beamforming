import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def dbm_to_watt(dbm):

    return 10 ** ((dbm - 30) / 10)


def cn_rand(shape, rng):

    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def solve_eq36_extended_multiuser(H, Gamma_lin, Pt_watt, Noise_C_watt):

    K, Nt = H.shape

    W = [cp.Variable((Nt, Nt), hermitian=True) for _ in range(K + 1)]

    RX = sum(W)

    U = cp.Variable((Nt, Nt), hermitian=True)

    I = np.eye(Nt)

    block = cp.bmat([

        [U, I],

        [I, RX]
    ])

    constraints = [

        block >> 0,

        cp.real(cp.trace(RX)) <= Pt_watt
    ]

    for i in range(K + 1):

        constraints.append(W[i] >> 0)

    for k in range(K):

        hk = H[k, :].reshape(-1, 1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        interference = 0

        for i in range(K + 1):

            if i != k:

                interference += cp.real(cp.trace(Qk @ W[i]))

        constraints.append(

            desired - Gamma_lin * interference >= Gamma_lin * Noise_C_watt
        )

    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(U))), constraints)

    try:

        prob.solve(solver=cp.MOSEK, verbose=False)

    except:

        prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, None, prob.status

    W_values = [Wi.value for Wi in W]

    RX_value = np.zeros((Nt, Nt), dtype=complex)

    for Wi in W_values:

        RX_value += Wi

    return W_values, RX_value, prob.status


def extended_target_mse_db(RX, Noise_R_watt, Nr, L, is_evd=False):

    RX = (RX + RX.conj().T) / 2

    eigvals, eigvecs = np.linalg.eigh(RX)

    if is_evd:

        eigvals = np.maximum(eigvals, 1e-4)

    else:

        eigvals = np.maximum(eigvals, 1e-8)

    RX_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.conj().T

    mse = (Noise_R_watt * Nr / L) * np.real(np.trace(RX_inv))

    return 10 * np.log10(mse + 1e-12)

def theorem4_rank1_recovery(W_values, H):

    K, Nt = H.shape

    RX_opt = np.zeros((Nt, Nt), dtype=complex)

    for W in W_values:

        RX_opt += W

    W_rank1_users = []

    for k in range(K):

        Wk = (W_values[k] + W_values[k].conj().T) / 2

        hk = H[k, :].reshape(-1, 1)

        denom = np.real(hk.conj().T @ Wk @ hk).item()

        if denom <= 1e-12:

            wk = np.zeros((Nt, 1), dtype=complex)

        else:

            wk = Wk @ hk / np.sqrt(denom)

        Wk_rank1 = wk @ wk.conj().T

        W_rank1_users.append(Wk_rank1)

    W_sum_users = np.zeros((Nt, Nt), dtype=complex)

    for Wk in W_rank1_users:

        W_sum_users += Wk

    W_radar_new = RX_opt - W_sum_users

    W_radar_new = (W_radar_new + W_radar_new.conj().T) / 2

    W_new = W_rank1_users + [W_radar_new]

    RX_new = np.zeros((Nt, Nt), dtype=complex)

    for W in W_new:

        RX_new += W

    return W_new, RX_new

def rank1_evd_approx(W_values):

    W_rank1_values = []

    for W in W_values:

        W = (W + W.conj().T) / 2

        eigvals, eigvecs = np.linalg.eigh(W)

        idx = np.argmax(eigvals)

        lambda_max = max(eigvals[idx], 0)

        u_max = eigvecs[:, idx].reshape(-1, 1)

        W_rank1 = lambda_max * (u_max @ u_max.conj().T)

        W_rank1_values.append(W_rank1)

    RX_rank1 = np.zeros_like(W_values[0], dtype=complex)

    for W1 in W_rank1_values:
        RX_rank1 += W1

    return W_rank1_values, RX_rank1


# ==============================
# Parameters
# ==============================

Nt = 16

Nr = 20

L = 30

Pt_dbm = 30

Pt_watt = dbm_to_watt(Pt_dbm)

Noise_C_dbm = 0

Noise_C_watt = dbm_to_watt(Noise_C_dbm)

Noise_R_dbm = 0

Noise_R_watt = dbm_to_watt(Noise_R_dbm)

K_list = np.arange(1, 13)

SINR_dB_list = [10, 20]

results = {
    10: {"convex": [], "proposed": [], "evd": []},
    20: {"convex": [], "proposed": [], "evd": []}
}


# ==============================
# Main loop
# ==============================

for SINR_dB in SINR_dB_list:

    Gamma_lin = 10 ** (SINR_dB / 10)

    print("\nSolving SINR =", SINR_dB, "dB")

    for K in K_list:

        print("K =", K)

        rng = np.random.default_rng(30245109 + 100 * SINR_dB + K)

        H = cn_rand((K, Nt), rng)

        W_values, RX_opt, status = solve_eq36_extended_multiuser(
            H=H,
            Gamma_lin=Gamma_lin,
            Pt_watt=Pt_watt,
            Noise_C_watt=Noise_C_watt
        )

        

        if status in ["optimal", "optimal_inaccurate"]:

            mse_convex_db = extended_target_mse_db(
                RX=RX_opt,
                Noise_R_watt=Noise_R_watt,
                Nr=Nr,
                L=L,
                is_evd=False
            )

            _, RX_evd = rank1_evd_approx(W_values)

            mse_evd_db = extended_target_mse_db(
                RX=RX_evd,
                Noise_R_watt=Noise_R_watt,
                Nr=Nr,
                L=L,
                is_evd=True
            )

            mse_proposed_db = mse_convex_db

            W_theorem4, RX_theorem4 = theorem4_rank1_recovery(W_values, H)

            mse_theorem4_db = extended_target_mse_db(

                RX = RX_theorem4,

                Noise_R_watt = Noise_R_watt,

                Nr = Nr,

                L = L,

                is_evd = False
            )

        else:

            mse_convex_db = np.nan

            mse_proposed_db = np.nan

            mse_evd_db = np.nan

        results[SINR_dB]["convex"].append(mse_convex_db)

        results[SINR_dB]["proposed"].append(mse_proposed_db)

        results[SINR_dB]["evd"].append(mse_evd_db)

        print("status =", status,
              "| convex =", mse_convex_db,
              "| proposed =", mse_proposed_db,
              "| evd =", mse_evd_db)
        
        print("Convex MSE =", mse_convex_db)
        
        print("Theorem 4 Rank-1 MSE =", mse_theorem4_db)


# ==============================
# Plot Fig. 7
# ==============================

plt.figure(figsize=(7, 5))

# SINR = 20 dB
plt.plot(K_list, results[20]["evd"], 'm--x',
         linewidth=2, label='Eigenvalue Decomposition, SINR = 20dB')

plt.plot(K_list, results[20]["convex"], 'k:',
         linewidth=2, label='Convex Relaxation Bound, SINR = 20dB')

plt.plot(K_list, results[20]["proposed"], 'ro',
         markerfacecolor='none', markersize=8,
         label='Proposed Rank-1 Optimum, SINR = 20dB')


# SINR = 10 dB
plt.plot(K_list, results[10]["evd"], 'y--d',
         linewidth=2, markerfacecolor='none',
         label='Eigenvalue Decomposition, SINR = 10dB')

plt.plot(K_list, results[10]["convex"], 'b:',
         linewidth=2, label='Convex Relaxation Bound, SINR = 10dB')

plt.plot(K_list, results[10]["proposed"], 'r^',
         markerfacecolor='none', markersize=8,
         label='Proposed Rank-1 Optimum, SINR = 10dB')


plt.xlabel('Number of Users')
plt.ylabel('MSE (dB)')
plt.grid(True)
plt.legend()
plt.xlim([1, 12])

plt.savefig("Figure07_Extended_Target_User_Number.png", dpi=600, bbox_inches="tight")
plt.show()