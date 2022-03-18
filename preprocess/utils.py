import math
import numpy as np
import cv2



# Estimate yaW angles according to facial landmarks
def estimate_yaw_angle(std_lm, pred_lm):
    '''
    std_lm: 3D landmarks of standard front face 
    pred_lm: 2D landmarks from landmark detector
    '''
    x = pred_lm
    X = std_lm
    X = np.reshape(X,[-1,3])
    _,R,_ = estimate_affine_matrix_3d22d(X, x)
    angles = matrix2angle(R)

    yaw_angle = angles[1]

    return yaw_angle


def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T
    x = x.T
    assert (x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert (n >= 4)

    # --- 1. normalization
    # 2d points
    mean = np.mean(x, 1)  # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x ** 2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean * scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1)  # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3, :] - X
    average_norm = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4, 4), dtype=np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean * scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n * 2, 8), dtype=np.float32)
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])

    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype=np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))

    # --- 5. convert to rigid parameters
    s, R, t = P2sRt(P_Affine)
    return s, R, t

def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert (isRotationMatrix)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi 
    return rx, ry, rz
    # return x, y, z

def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6