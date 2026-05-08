import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# =====================================================
# Basic helper functions
# =====================================================

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


# =====================================================
# Proposed CRB-Min Design: Eq. (33)
# =====================================================

def solve_eq33_multiuser(H, a0, a0_dot, b0, b0_dot, Gamma_lin, Pt_watt, Noise_C_watt):

    K, Nt = H.shape

    W = [cp.Variable((Nt, Nt), hermitian=True) for _ in range(K)]

    t = cp.Variable(nonneg=True)

    RX = sum(W)

    A = b0 @ a0.conj().T

    A_dot = b0_dot @ a0.conj().T + b0 @ a0_dot.conj().T

    M11 = cp.real(cp.trace(A_dot.conj().T @ A_dot @ RX)) - t

    M12 = cp.trace(A_dot.conj().T @ A @ RX)

    M21 = cp.trace(A.conj().T @ A_dot @ RX)

    M22 = cp.real(cp.trace(A.conj().T @ A @ RX))

    M = cp.bmat([

        [cp.reshape(M11, (1, 1), order="C"), cp.reshape(M12, (1, 1), order="C")],

        [cp.reshape(M21, (1, 1), order="C"), cp.reshape(M22, (1, 1), order="C")]
    ])

    constraints = [

        M >> 0,

        cp.real(cp.trace(RX)) <= Pt_watt
    ]

    for k in range(K):

        constraints.append(W[k] >> 0)

    for k in range(K):

        hk = H[k, :].reshape(-1, 1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        interference = 0

        for i in range(K):

            if i != k:

                interference += cp.real(cp.trace(Qk @ W[i]))

        constraints.append(

            desired - Gamma_lin * interference >= Gamma_lin * Noise_C_watt
        )

    prob = cp.Problem(cp.Maximize(t), constraints)

    try:

        prob.solve(solver=cp.MOSEK, verbose=False)

    except Exception:

        prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, prob.status

    RX_value = np.zeros((Nt, Nt), dtype=complex)

    for Wi in W:

        RX_value += Wi.value

    return RX_value, prob.status


# =====================================================
# Radar-only reference beampattern: Eq. (10)
# =====================================================

def solve_radar_only_eq10(Nt, angle_grid, Pt_watt, theta0=0, theta1=-5, theta2=5):

    R = cp.Variable((Nt, Nt), hermitian=True)

    t = cp.Variable()

    a0 = steering_vector_centered(Nt, theta0)

    a1 = steering_vector_centered(Nt, theta1)

    a2 = steering_vector_centered(Nt, theta2)

    P0 = cp.real(a0.conj().T @ R @ a0)

    P1 = cp.real(a1.conj().T @ R @ a1)

    P2 = cp.real(a2.conj().T @ R @ a2)

    constraints = [

        R >> 0,

        cp.diag(R) == (Pt_watt / Nt) * np.ones(Nt),

        P1 == P0 / 2,

        P2 == P0 / 2
    ]

    for theta in angle_grid:

        if abs(theta) > 5:

            a_theta = steering_vector_centered(Nt, theta)

            Ptheta = cp.real(a_theta.conj().T @ R @ a_theta)

            constraints.append(P0 - Ptheta >= t)

    prob = cp.Problem(cp.Minimize(-t), constraints)

    try:

        prob.solve(solver=cp.MOSEK, verbose=False)

    except Exception:
        prob.solve(solver=cp.CLARABEL, verbose=False)


    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, prob.status

    return R.value, prob.status


def make_reference_pattern(R_ref, angle_grid, Nt):

    Pd_ref = []

    for theta in angle_grid:

        a_theta = steering_vector_centered(Nt, theta)

        power = np.real(a_theta.conj().T @ R_ref @ a_theta).item()

        Pd_ref.append(power)

    Pd_ref = np.array(Pd_ref)

    Pd_ref = Pd_ref / np.max(Pd_ref)

    return Pd_ref


# =====================================================
# Design 2 [12]: SINR-constrained beampattern approximation
# =====================================================

def solve_design2_sinr_constrained(H, angle_grid, Pd, Gamma_lin, Pt_watt, Noise_C_watt):

    K, Nt = H.shape

    W = [cp.Variable((Nt, Nt), hermitian=True) for _ in range(K)]

    alpha = cp.Variable(nonneg=True)

    RX = sum(W)

    obj_terms = []

    for m, theta in enumerate(angle_grid):

        a_theta = steering_vector_centered(Nt, theta)

        p_theta = cp.real(a_theta.conj().T @ RX @ a_theta)

        obj_terms.append(cp.square(alpha * Pd[m] - p_theta))

    constraints = [

        cp.real(cp.trace(RX)) <= Pt_watt
    ]

    for k in range(K):

        constraints.append(W[k] >> 0)

    for k in range(K):

        hk = H[k, :].reshape(-1, 1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        interference = 0

        for i in range(K):

            if i != k:

                interference += cp.real(cp.trace(Qk @ W[i]))

        constraints.append(

            desired - Gamma_lin * interference >= Gamma_lin * Noise_C_watt
        )

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

    try:

        prob.solve(solver=cp.MOSEK, verbose=False)

    except Exception:

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)

        except Exception:
         
         return None, "solver_failed"



    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, prob.status

    RX_value = np.zeros((Nt, Nt), dtype=complex)
    for Wi in W:
        RX_value += Wi.value

    return RX_value, prob.status


# =====================================================
# Design 1 [11]: communication + dedicated radar covariance
# =====================================================

def solve_design1_joint_radar_comm(
    H,
    angle_grid,
    Pd,
    Gamma_lin,
    Pt_watt,
    Noise_C_watt,
    radar_power_ratio=0.05
):

    K, Nt = H.shape

    W = [cp.Variable((Nt, Nt), hermitian=True) for _ in range(K)]
    Rr = cp.Variable((Nt, Nt), hermitian=True)
    alpha = cp.Variable(nonneg=True)

    RX = sum(W) + Rr

    obj_terms = []

    for m, theta in enumerate(angle_grid):
        a_theta = steering_vector_centered(Nt, theta)
        p_theta = cp.real(a_theta.conj().T @ RX @ a_theta)
        obj_terms.append(cp.square(alpha * Pd[m] - p_theta))

    constraints = [
        Rr >> 0,
        cp.real(cp.trace(RX)) <= Pt_watt,
        cp.diag(Rr) == (radar_power_ratio * Pt_watt / Nt) * np.ones(Nt)
    ]

    for k in range(K):
        constraints.append(W[k] >> 0)

    for k in range(K):
        hk = H[k, :].reshape(-1, 1)
        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        comm_interference = 0
        for i in range(K):
            if i != k:
                comm_interference += cp.real(cp.trace(Qk @ W[i]))

        radar_interference = cp.real(cp.trace(Qk @ Rr))

        constraints.append(
            desired
            - Gamma_lin * comm_interference
            - Gamma_lin * radar_interference
            >= Gamma_lin * Noise_C_watt
        )

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)

    except Exception:

        try:

            prob.solve(solver=cp.CLARABEL, verbose=False)

        except Exception:

            return None, "solver_failed"

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, prob.status

    RX_value = np.zeros((Nt, Nt), dtype=complex)

    for Wi in W:
        RX_value += Wi.value

    RX_value += Rr.value

    return RX_value, prob.status


# =====================================================
# Root-CRB calculation
# =====================================================

def root_crb_from_RX(RX, a0, a0_dot, b0, b0_dot, L, Noise_R_watt, alpha=1.0):

    if RX is None:
        return np.nan

    A = b0 @ a0.conj().T

    A_dot = b0_dot @ a0.conj().T + b0 @ a0_dot.conj().T

    t1 = np.real(np.trace(A_dot.conj().T @ A_dot @ RX))

    t2 = np.abs(np.trace(A_dot.conj().T @ A @ RX)) ** 2

    t3 = np.real(np.trace(A.conj().T @ A @ RX))

    fisher = t1 - t2 / t3

    if fisher <= 0:

        return np.nan

    crb = Noise_R_watt / (2 * np.abs(alpha) ** 2 * L * fisher)

    return np.sqrt(crb) * 180 / np.pi


# =====================================================
# Main parameters
# =====================================================

Nt = 16

Nr = 20

L = 30

Pt_dbm = 30

Pt_watt = 10 ** ((Pt_dbm - 30) / 10)

Noise_C_dbm = 0

Noise_C_watt = 10 ** ((Noise_C_dbm - 30) / 10)

Noise_R_dbm = 0

Noise_R_watt = 10 ** ((Noise_R_dbm - 30) / 10)

theta0 = 0

a0 = steering_vector_centered(Nt, theta0)

b0 = steering_vector_centered(Nr, theta0)

a0_dot = steering_derivative_centered(Nt, theta0)

b0_dot = steering_derivative_centered(Nr, theta0)

angle_grid = np.linspace(-90, 90, 721)

Gamma_dB_list = np.arange(2, 22, 2)

K_list = [6, 14]

MC = 10  # increase to 20 or 30 for smoother curves


# =====================================================
# Reference beampattern
# =====================================================

R_ref, status_ref = solve_radar_only_eq10(
    Nt=Nt,
    angle_grid=angle_grid,
    Pt_watt=Pt_watt,
    theta0=0,
    theta1=-5,
    theta2=5
)

print("Eq.(10) status =", status_ref)

Pd = make_reference_pattern(
    R_ref=R_ref,
    angle_grid=angle_grid,
    Nt=Nt
)


# =====================================================
# Monte Carlo sweep
# =====================================================

results = {}

for K_now in K_list:

    print("\nSolving for K =", K_now)

    crb_prop_list = []

    crb_d1_list = []

    crb_d2_list = []

    for Gamma_dB in Gamma_dB_list:

        print("\nK =", K_now, "| Gamma =", Gamma_dB, "dB")

        Gamma_lin = 10 ** (Gamma_dB / 10)

        prop_mc = []

        d1_mc = []

        d2_mc = []


        for mc in range(MC):

            print("  MC =", mc + 1, "/", MC)

            rng_mc = np.random.default_rng(

                30245109 + 100000 * K_now + 1000 * int(Gamma_dB) + mc
            )

            H_now = cn_rand((K_now, Nt), rng_mc)

            RX_prop, status_prop = solve_eq33_multiuser(

                H = H_now,

                a0 = a0,

                a0_dot = a0_dot,

                b0 = b0,

                b0_dot = b0_dot,

                Gamma_lin = Gamma_lin,

                Pt_watt = Pt_watt,

                Noise_C_watt = Noise_C_watt
            )

            RX_d2, status_d2 = solve_design2_sinr_constrained(

                H = H_now,

                angle_grid = angle_grid,

                Pd = Pd,

                Gamma_lin = Gamma_lin,

                Pt_watt = Pt_watt,

                Noise_C_watt = Noise_C_watt
            )

            RX_d1, status_d1 = solve_design1_joint_radar_comm(
                H = H_now,
                angle_grid = angle_grid,
                Pd = Pd,
                Gamma_lin = Gamma_lin,
                Pt_watt = Pt_watt,
                Noise_C_watt = Noise_C_watt,
                radar_power_ratio = 0.1
            )

            crb_prop = root_crb_from_RX(
                RX_prop, a0, a0_dot, b0, b0_dot, L, Noise_R_watt
            )

            crb_d2 = root_crb_from_RX(
                RX_d2, a0, a0_dot, b0, b0_dot, L, Noise_R_watt
            )

            crb_d1 = root_crb_from_RX(
                RX_d1, a0, a0_dot, b0, b0_dot, L, Noise_R_watt
            )

            prop_mc.append(crb_prop)

            d2_mc.append(crb_d2)

            d1_mc.append(crb_d1)

            print("    status:", status_prop, status_d2, status_d1)

            print("    CRB:", crb_prop, crb_d2, crb_d1)

        crb_prop_avg = np.nanmean(prop_mc)

        crb_d2_avg = np.nanmean(d2_mc)

        crb_d1_avg = np.nanmean(d1_mc)

        crb_prop_list.append(crb_prop_avg)

        crb_d2_list.append(crb_d2_avg)

        crb_d1_list.append(crb_d1_avg)

        print("Average Proposed =", crb_prop_avg)

        print("Average Design 2 =", crb_d2_avg)

        print("Average Design 1 =", crb_d1_avg)

    results[K_now] = {

    "prop": np.array(crb_prop_list),

    "d2": np.array(crb_d2_list),

    "d1": np.array(crb_d1_list)

        }


# =====================================================
# Plot Fig. 5
# =====================================================

plt.figure(figsize=(7, 5))

plt.plot(

    Gamma_dB_list,

    results[14]["d1"],

    "b--x",

    linewidth=2,

    label="Beampattern Approx. Design 1 [11], K = 14"
)

plt.plot(

    Gamma_dB_list,

    results[14]["d2"],

    "r--^",

    linewidth=2,

    label="Beampattern Approx. Design 2 [12], K = 14"
)

plt.plot(

    Gamma_dB_list,

    results[14]["prop"],

    "--o",

    color="orange",

    linewidth=2,

    label="Proposed CRB-Min Design, K = 14"
)

########## K = 6 solid ###########################
plt.plot(

    Gamma_dB_list,

    results[6]["d1"],

    "b-x",

    linewidth=2,

    label="Beampattern Approx. Design 1 [11], K = 6"
)

plt.plot(

    Gamma_dB_list,

    results[6]["d2"],

    "r-^",

    linewidth=2,

    label="Beampattern Approx. Design 2 [12], K = 6"
)

plt.plot(

    Gamma_dB_list,

    results[6]["prop"],

    "-o",

    color="orange",

    linewidth=2,

    label="Proposed CRB-Min Design, K = 6"
)

plt.xlabel("SINR (dB)")

plt.ylabel("Root-CRB (deg)")

plt.grid(True)

plt.legend()

plt.xlim([6, 20])


plt.savefig("Figure05_Radar_Comm_Tradeoff_MC.png", dpi=600, bbox_inches="tight")

plt.show()