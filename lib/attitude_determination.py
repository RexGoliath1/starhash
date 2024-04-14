import numpy as np
from transform_utils import quaternion_to_dcm, dcm_to_quaternion

def quest(v_b, v_i, w = None, tolerance = 1e-5):
    """Quaternion Estimator (QUEST) algorithm. Return T_i^b (inertial to body, or 2nd frame to 1st) """
    if w is None:
        w = np.ones(v_b.shape[1])

    # B matrix computation
    B = v_b @ np.diag(w) @ v_i.T
    S = B + B.T
    sigma = np.trace(B)

    # Helper variables
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    delta = np.linalg.det(S)
    kappa = np.trace(delta * np.linalg.inv(S))

    # Coefficients for the characteristic equation
    a = sigma**2 - kappa
    b = sigma**2 + Z.T @ Z
    c = delta + Z.T @ S @ Z
    d = Z.T @ S @ S @ Z 
    constant = a*b + c*sigma - d

    # Newton-Raphson method to find the root
    lambda_ = np.sum(w)
    last_lambda = 0.0
    while np.abs(lambda_ - last_lambda) >= tolerance:
        last_lambda = lambda_
        f = lambda_**4 - (a + b)*lambda_**2 - c*lambda_ + constant
        f_dot = 4*lambda_**3 - 2*(a + b)*lambda_ - c
        lambda_ -= f / f_dot

    # Optimal quaternion computation
    omega = lambda_
    alpha = omega**2 - sigma**2 + kappa
    beta = omega - sigma
    gamma = (omega + sigma)*alpha - delta
    X = (alpha*np.eye(3) + beta*S + S @ S) @ Z
    q_opt = np.hstack((X, gamma)) / np.sqrt(gamma**2 + np.linalg.norm(X)**2)

    # Convert quaternion to DCM
    C_opt = quaternion_to_dcm(q_opt)

    return C_opt, q_opt

def davenport(v_b, v_i, w = None):
    # Matrix K (4x4)
    if w is None:
        w = np.ones(v_b.shape[1])

    B = v_b @ np.diag(w) @ v_i.T
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    Z = Z.reshape(3,1)
    K = np.block([[B + B.T - np.eye(3) * np.trace(B), Z],
                  [Z.T, np.trace(B)]])

    # Eigenvector associated with largest eigenvalue
    evals, evecs = np.linalg.eig(K)
    idx = np.argmax(evals)
    q_opt = evecs[:, idx]
    C_opt = quaternion_to_dcm(q_opt);

    return C_opt, q_opt

def triad(b1, b2, r1, r2):
    """Tri-axial Attitude Determination (TRIAD)"""

    # Calculate v2
    v2 = np.cross(r1, r2) / np.linalg.norm(np.cross(r1, r2))

    # Calculate w2
    w2 = np.cross(b1, b2) / np.linalg.norm(np.cross(b1, b2))

    # Calculate rotation matrix C
    C_opt = np.outer(b1, r1) + np.outer(np.cross(b1, w2), np.cross(r1, v2)) + np.outer(w2, v2)

    # Convert rotation matrix to quaternion
    q_opt = dcm_to_quaternion(C_opt)

    return C_opt, q_opt

def foma(v_b, v_i, w, tolerance = 1e-5):
    if w is None:
        w = np.ones(v_b.shape[1])

    B = v_b @ np.diag(w) @ v_i.T
    det_B = np.linalg.det(B)
    adj_B = det_B * np.linalg.inv(B)
    norm_B = np.linalg.norm(B, 'fro')
    norm_adj_B = np.linalg.norm(adj_B, 'fro')

    lambda_ = np.sum(w)
    last_lambda = 0.0
    while abs(lambda_ - last_lambda) >= tolerance:
        last_lambda = lambda_

        f = (lambda_**2 - norm_B**2)**2 - 8*lambda_*det_B - 4*norm_adj_B**2
        f_dot = 4*lambda_*(lambda_**2 - norm_B**2) - 8*det_B
        lambda_ = lambda_ - f / f_dot

    kappa = 0.5 * (lambda_**2 - norm_B**2)
    zeta = kappa * lambda_ - det_B

    C_opt = ((kappa + norm_B**2) * B + lambda_ * adj_B.T - np.dot(B, B.T) @ B) / zeta
    q_opt = dcm_to_quaternion(C_opt)
    
    return C_opt, q_opt

def svd(v_b, v_i, w = None):
    if w is None:
        w = np.ones(v_b.shape[1])

    B = v_b @ np.diag(w) @ v_i.T
    U, _, Vt = np.linalg.svd(B)
    d = np.linalg.det(U) * np.linalg.det(Vt.T)

    C_opt = np.dot(U, np.diag([1, 1, d])) @ Vt
    q_opt = dcm_to_quaternion(C_opt)

    return C_opt, q_opt

def esoq2(v_b, v_i, w = None):
    # TODO: In progress
    if w is None:
        w = np.ones(v_b.shape[1])

    # Attitude profile matrix
    B = v_b @ np.diag(w) @ v_i.T
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    Z = Z.reshape(3,1)
    S = B + B.T - np.eye(3) * np.trace(B)

    K = np.block([[S, Z],
                  [Z.T, np.trace(B)]])

    adj_BBt = np.linalg.det(B + B.T) * np.linalg.inv(B + B.T)
    adj_K = np.linalg.det(K) * np.linalg.inv(K)

    b = -2 * np.trace(B)**2 + np.trace(adj_BBt) - Z.T @ Z
    c = np.trace(adj_K)
    d = np.linalg.det(K)
    p = (b/3)**2 + 4*d/3
    q = (b/3)**3 - 4*d*b/3 + (c**2)/2
    u1 = 2 * np.sqrt(p) * np.cos((1/3) * np.arccos(q / p**1.5)) + b/3

    # Calculate lambda for different cases
    if v_b.shape[1] == 2:
        g3 = np.sqrt(2 * np.sqrt(d) - b)
        g4 = np.sqrt(-2 * np.sqrt(d) - b)
        lambda_ = (g3 + g4) / 2
    else:
        g1 = np.sqrt(u1 - b)
        g2 = -2 * np.sqrt(u1**2 - 4 * d)
        lambda_ = 0.5 * (g1 + np.sqrt(g1 - b - g2))

    t = np.trace(B) - lambda_
    S = B + B.T - (np.trace(B) + lambda_) * np.eye(3)
    M = t * S - np.outer(Z, Z)

    e1 = np.cross(M[:, 1], M[:, 2])
    e2 = np.cross(M[:, 2], M[:, 0])
    e3 = np.cross(M[:, 0], M[:, 1])
    e = [e1, e2, e3][np.argmax([np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)])]

    e /= np.linalg.norm(e)
    x = np.vstack([Z, t])
    y = -np.vstack([S, Z.T]) @ e
    k = np.argmax(np.abs(x))
    h = np.sqrt(x[k] ** 2 + y[k] ** 2)
    sph = x[k]
    cph = y[k]

    # Calculate the quaternion
    q = np.array([sph * e[0], sph * e[1], sph * e[2], [cph]])
    q /= np.linalg.norm(q)

    # Convert quaternion to rotation matrix
    C = quaternion_to_dcm(q)

    return C, q
