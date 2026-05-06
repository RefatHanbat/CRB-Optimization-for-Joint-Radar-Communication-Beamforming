from f_closed_form_numerical_solutions_for_single_user import * 


def solve_eq33_multiuser(H, a0, a0_dot, b0, b0_dot, Gamma_lin, Pt_watt, Noise_C_watt):

    K, Nt = H.shape

    W = [cp.Variable((Nt,Nt), hermitian = True) for _ in range(K)]

    t = cp.Variable(nonneg = True)

    RX = sum(W)

    A = b0 @ a0.conj().T

    A_dot = b0_dot @ a0.conj().T + b0 @ a0_dot.conj().T

    M11 = cp.real(cp.trace(A_dot.conj().T @ A_dot @ RX)) - t

    M12 = cp.trace(A_dot.conj().T @ A @ RX)

    M21 = cp.trace(A.conj().T @ A_dot @ RX)

    M22 = cp.real(cp.trace(A.conj().T @ A @ RX))

    M = cp.bmat([

        [cp.reshape(M11, (1,1), order = "C"), cp.reshape(M12, (1,1), order = "C")],

        [cp.reshape(M21, (1,1), order = "C"), cp.reshape(M22, (1,1), order = "C")]
    ])

    constraints = [

        M >>0,

        cp.real(cp.trace(RX)) <= Pt_watt
    ]

    for k in range(K):

        constraints.append(W[k] >>0)

    for k in range(K):

        hk = H[k, :].reshape(-1,1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        interference = 0

        for i in range(K):

            if i != k:

                interference += cp.real(cp.trace(Qk @ W[i]))

        constraints.append(desired - Gamma_lin * interference >= Gamma_lin * Noise_C_watt)
    
    prob = cp.Problem(cp.Maximize(t), constraints)

    try:
        prob.solve(solver = cp.MOSEK, verbose = False)

    except:
        prob.solve(solver = cp.CLARABEL, verbose = False)

    RX_value = sum([Wi.value for Wi in W])

    return RX_value, t.value, prob.status

def desired_beam_3db(angle_grid, beamwidth = 10):

    Pd = np.zeros_like(angle_grid, dtype=float)

    Pd[np.abs(angle_grid) <= beamwidth / 2] = 1.0

    return Pd

def desired_beampattern(angle_grid):

    Pd = np.zeros_like(angle_grid)

    Pd[np.abs(angle_grid) <= 5] = 1

    return Pd

def solve_design2_eq9(angle_grid, Pd, Pt_watt):

    R = cp.Variable((Nt, Nt), hermitian=True)

    alpha = cp.Variable(nonneg=True)

    obj_terms = []

    for m, theta in enumerate(angle_grid):

        a_theta = steering_vector_centered(Nt, theta)

        p_theta = cp.real(a_theta.conj().T @ R @ a_theta)

        obj_terms.append(cp.square(alpha * Pd[m] - p_theta))

    constraints = [

        R >> 0,

        cp.diag(R) == (Pt_watt / Nt) * np.ones(Nt)

    ]

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

    try:

        prob.solve(solver=cp.MOSEK, verbose=False)

    except:

        prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        
        return None, None, prob.status

    return R.value, alpha.value, prob.status

def solve_radar_only_eq10(Nt, angle_grid, Pt_watt, theta0 = 0, theta1 = -5, theta2 = 5):

    R = cp.Variable((Nt,Nt), hermitian = True)

    t = cp.Variable()

    a0 = steering_vector_centered(Nt, theta0)

    a1 = steering_vector_centered(Nt, theta1)

    a2 = steering_vector_centered(Nt, theta2)

    P0 = cp.real(a0.conj().T @ R @ a0)

    P1 = cp.real(a1.conj().T @ R @ a1)

    P2 = cp.real(a2.conj().T @ R @ a2)

    constraints = [

        R >>0,

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

        prob.solve(solver = cp.MOSEK, verbose = False)

    except:

        prob.solve(solver = cp.CLARABEL, verbose = False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, None, prob.status
    
    return R.value, t.value, prob.status

def solve_design1_joint_radar_comm(H, angle_grid, Pd, Gamma_lin, Pt_watt, Noise_C_watt, radar_power_ratio = 0.3):

    K, Nt = H.shape

    W = [cp.Variable((Nt, Nt), hermitian = True) for _ in range(K)]

    Rr = cp.Variable((Nt, Nt), hermitian = True)

    alpha = cp.Variable(nonneg = True)

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

        hk = H[k, :].reshape(-1,1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        comm_interference = 0

        for i in range(K):

            if i != k:

                comm_interference += cp.real(cp.trace(Qk @ W[i]))

        radar_interference = cp.real(cp.trace(Qk @ Rr))

        constraints.append(

            desired - Gamma_lin * comm_interference - Gamma_lin * radar_interference >= Gamma_lin * Noise_C_watt
        )

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

    try:

        prob.solve(solver = cp.MOSEK, verbose = False)

    except:

        prob.solve(solver = cp.CLARABEL, verbose = False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:

        return None, prob.status
    
    RX_value = np.zeros((Nt,Nt), dtype = complex)

    for Wi in W:

        RX_value += Wi.value

    RX_value += Rr.value

    return RX_value, prob.status


def compute_beampattern_db(RX, angle_grid, Nt):

    bp = []

    for theta in angle_grid:

        a_theta = steering_vector_centered(Nt, theta)

        power = np.real(a_theta.conj().T @ RX @ a_theta).item()

        bp.append(max(power, 1e-12))

    bp = np.array(bp)

    return 10 * np.log10(bp)



### parametes ####

K = 4

Gamma_dB = 15

Gamma_lin = 10 **(Gamma_dB / 10)

H = cn_rand((K,Nt),rng)

theta0 = 0

a0 = steering_vector_centered(Nt, theta0)

b0 = steering_vector_centered(Nr, theta0)

a0_dot = steering_derivative_centered(Nt, theta0)

b0_dot = steering_derivative_centered(Nr, theta0)


angle_grid = np.linspace(-90,90,721)

RX_crb, t_crb, status = solve_eq33_multiuser(H = H, a0 = a0, a0_dot = a0_dot, b0 = b0, b0_dot = b0_dot, Gamma_lin = Gamma_lin,
                              Pt_watt = Pt_watt, Noise_C_watt = Noise_C_watt)


R_ref, t_ref, status_ref = solve_radar_only_eq10(

    Nt = Nt,

    angle_grid = angle_grid,

    Pt_watt = Pt_watt,

    theta0 = 0,

    theta1 = -5,

    theta2 = 5
)
Pd_ref = []

for theta in angle_grid:

    a_theta = steering_vector_centered(Nt, theta)

    power = np.real(a_theta.conj().T @ R_ref @ a_theta).item()

    Pd_ref.append(power)

Pd_ref = np.array(Pd_ref)

Pd_ref = Pd_ref / np.max(Pd_ref)

Pd = Pd_ref

R_design2, alpha, status = solve_design2_eq9(

    angle_grid = angle_grid,

    Pd = Pd,

    Pt_watt = Pt_watt
)

bp_design2 = []

for theta in angle_grid:

    a_theta = steering_vector_centered(Nt, theta)

    power = np.real(

        a_theta.conj().T @ R_design2 @ a_theta
    ).item()

    bp_design2.append(power)

bp_design2 = np.array(bp_design2)

bp_design2_db = 10 * np.log10(bp_design2 + 1e-12)

bp_crb = compute_beampattern_db(RX_crb, angle_grid, Nt)


####### Design 1 #################

RX_d1, status_d1 = solve_design1_joint_radar_comm(H = H, angle_grid = angle_grid,
                                                  
                                                  Pd = Pd,

                                                  Gamma_lin = Gamma_lin,

                                                  Pt_watt = Pt_watt,

                                                  Noise_C_watt = Noise_C_watt,
                                                  
                                                  radar_power_ratio = 0.3 )

bp_d1 = compute_beampattern_db(RX_d1, angle_grid, Nt)


plt.figure(figsize=(6,4))

plt.plot(angle_grid, bp_crb, color = 'orange', linewidth = 2, label = 'Proposed CRB-Min Design')

plt.plot(angle_grid, bp_design2_db, 'g-.', linewidth =2, label = 'Beampattern Approx. Design 2')

plt.plot(angle_grid, bp_d1, 'b:', linewidth = 2.5, label = 'Beampattern Approx. Design 1')


plt.xlabel('Angle (deg)')

plt.ylabel('Beampattern (dBi)')

plt.grid(True)

plt.legend()

plt.xlim([-90,90])

plt.savefig("Figure03_Beampattern.png", dpi=600, bbox_inches='tight')

plt.show()