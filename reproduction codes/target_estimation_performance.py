from f_closed_form_numerical_solutions_for_single_user import * 

from Beampattern_point_targets import *

def point_target_crb(RX, A, A_dot, alpha, L, Noise_R_watt):

    t1 = np.real(np.trace(A_dot.conj().T @ A_dot @ RX))

    t2 = np.abs(
        np.trace(A_dot.conj().T @ A @ RX)
    )**2

    t3 = np.real(

        np.trace(A.conj().T @ A @ RX)
    )

    fisher = t1 - t2 / t3

    crb_rad = Noise_R_watt / (2 * (np.abs(alpha)**2) * L * fisher)

    crb_deg = np.rad2deg(np.sqrt(crb_rad))

    return crb_deg

snr_dB_list = np.arange(-20,1,2)

A0 = b0 @ a0.conj().T

A0_dot = b0_dot @ a0.conj().T + b0 @ a0_dot.conj().T

crb_prop = []

crb_d1 = []

crb_d2 = []

for snr_db in snr_dB_list:

    snr_lin = 10**(snr_db / 10)

    alpha = np.sqrt(

        snr_lin * Noise_R_watt / (L * Pt_watt)
    )

    crb_prop.append(

        point_target_crb(RX_crb,A0, A0_dot, alpha, L, Noise_R_watt )
    )

    crb_d1.append(

        point_target_crb(RX_d1, A0, A0_dot, alpha, L, Noise_R_watt)
    )

    crb_d2.append(

        point_target_crb(

            R_design2, A0, A0_dot, alpha, L, Noise_R_watt
        )
    )

search_grid = np.linspace(-90, 90, 721)

def estimate_theta_mle(Y, search_grid):

    scores = []

    for theta in search_grid:

        A = (

            steering_vector_centered(Nr, theta)

            @

            steering_vector_centered(Nt, theta).conj().T
        )

        score = np.linalg.norm(A.conj().T @ Y)**2

        scores.append(score)

    idx = np.argmax(scores)

    return search_grid[idx]

def generate_waveform_from_cov(RX, L, rng):

    eigvals, eigvecs = np.linalg.eigh((RX + RX.conj().T) / 2)

    eigvals = np.maximum(eigvals, 0)

    RX_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    S = cn_rand((RX.shape[0], L), rng)

    X = RX_sqrt @ S

    return X


def estimate_theta_mle(Y, X, search_grid):

    best_score = -np.inf

    best_theta = None

    for theta in search_grid:

        a_theta = steering_vector_centered(Nt, theta)

        b_theta = steering_vector_centered(Nr, theta)

        A_theta = b_theta @ a_theta.conj().T

        AX = A_theta @ X

        numerator = np.abs(np.vdot(AX, Y))**2

        denominator = np.linalg.norm(AX, "fro")**2 + 1e-12

        score = numerator / denominator

        if score > best_score:

            best_score = score

            best_theta = theta

    return best_theta


def monte_carlo_mle_rmse(RX, snr_dB_list, theta0, L, Pt_watt, Noise_R_watt, MC=200):

    rmse_list = []


    a0 = steering_vector_centered(Nt, theta0)

    b0 = steering_vector_centered(Nr, theta0)

    A0 = b0 @ a0.conj().T

    search_grid = np.linspace(-90, 90, 721)

    for snr_db in snr_dB_list:

        snr_lin = 10 ** (snr_db / 10)

        alpha = np.sqrt(snr_lin * Noise_R_watt / (L * Pt_watt))

        errors = []

        for mc in range(MC):

            X = generate_waveform_from_cov(RX, L, rng)

            N = np.sqrt(Noise_R_watt) * cn_rand((Nr, L), rng)

            Y = alpha * A0 @ X + N

            theta_hat = estimate_theta_mle(Y, X, search_grid)

            errors.append(theta_hat - theta0)

        rmse = np.sqrt(np.mean(np.array(errors)**2))

        rmse_list.append(rmse)

        print("SNR =", snr_db, "dB | RMSE =", rmse)

    return np.array(rmse_list)


rmse_prop = monte_carlo_mle_rmse(

    RX=RX_crb,

    snr_dB_list=snr_dB_list,

    theta0=theta0,

    L=L,

    Pt_watt=Pt_watt,

    Noise_R_watt=Noise_R_watt,

    MC=200
)

rmse_d1 = monte_carlo_mle_rmse(

    RX=RX_d1,

    snr_dB_list=snr_dB_list,

    theta0=theta0,

    L=L,

    Pt_watt=Pt_watt,

    Noise_R_watt=Noise_R_watt,

    MC=200
)

rmse_d2 = monte_carlo_mle_rmse(

    RX=R_design2,

    snr_dB_list=snr_dB_list,

    theta0=theta0,

    L=L,

    Pt_watt=Pt_watt,

    Noise_R_watt=Noise_R_watt,

    MC=200
)


plt.figure(figsize=(7,5))

plt.semilogy(snr_dB_list, rmse_d1, 'b--', linewidth=2, label='Beampattern Approx. Design 1 [11], MLE')

plt.semilogy(snr_dB_list, crb_d1, 'b-', linewidth=2, label='Beampattern Approx. Design 1 [11], CRB')

plt.semilogy(snr_dB_list, rmse_d2, 'r--', linewidth=2, label='Beampattern Approx. Design 2 [12], MLE')

plt.semilogy(snr_dB_list, crb_d2, 'r-', linewidth=2, label='Beampattern Approx. Design 2 [12], CRB')

plt.semilogy(snr_dB_list, rmse_prop, '--', color='orange', linewidth=2, label='Proposed CRB-Min Design, MLE')

plt.semilogy(snr_dB_list, crb_prop, '-', color='orange', linewidth=2, label='Proposed CRB-Min Design, CRB')

plt.xlabel('Radar SNR (dB)')

plt.ylabel('RMSE (deg)')

plt.grid(True, which='both')

plt.legend()

plt.xlim([-20, 0])

plt.ylim([1e-1, 1e2])


plt.savefig("Figure04_RMSE_CRB.png", dpi=600, bbox_inches="tight")

plt.show()