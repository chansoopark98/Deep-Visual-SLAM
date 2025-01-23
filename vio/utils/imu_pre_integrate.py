import numpy as np

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Returns the skew-symmetric matrix of a vector v.
    For v = [x, y, z], skew_symmetric(v) =
        [ 0  -z   y]
        [ z   0  -x]
        [-y   x   0]
    """
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ], dtype=np.float64)

def so3_exponential(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) vector (3x1) to SO(3) rotation matrix (3x3).
    Uses Rodrigues' formula for small rotation vectors.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        # Near zero rotation => approximate by identity + skew
        return np.eye(3) + skew_symmetric(omega)
    axis = omega / theta
    K = skew_symmetric(axis)
    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )

class IMUPreintegrator:
    """
    Implements IMU pre-integration over multiple small time steps.
    Accumulates the relative rotation (dR), relative velocity (dV),
    and relative position (dP) between two frames, plus the covariance.

    Based on the method in:
      - Forster et al., "Preintegration on Manifold for Efficient
        Visual-Inertial Bundle Adjustment" (2015, 2017)
    """
    def __init__(self,
                 acc_bias: np.ndarray,
                 gyro_bias: np.ndarray,
                 acc_noise: float,
                 gyro_noise: float,
                 acc_random_walk: float,
                 gyro_random_walk: float,
                 gravity: np.ndarray = np.array([0, 0, -9.81])):
        """
        Parameters:
        -----------
        acc_bias, gyro_bias : (3,) np.ndarray
            Initial estimates of accelerometer / gyroscope biases.
        acc_noise, gyro_noise : float
            Std dev of IMU measurement noise (white noise).
            [Typically in (m/s^2)/√Hz for acc, (rad/s)/√Hz for gyro]
        acc_random_walk, gyro_random_walk : float
            Std dev of bias random walk for accelerometer, gyro.
        gravity : (3,) np.ndarray
            Gravity vector in the navigation frame, default = [0, 0, -9.81].
        """
        self.acc_bias = acc_bias.copy()
        self.gyro_bias = gyro_bias.copy()
        self.gravity = gravity.copy()

        # Noise parameters
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        self.acc_rw = acc_random_walk
        self.gyro_rw = gyro_random_walk

        # Pre-integrated quantities
        self.delta_R = np.eye(3)  # rotation from frame_i to frame_j
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)

        # Covariance matrix for [rot-err, vel-err, pos-err, ba-err, bg-err]
        # Typically dimension = 15 x 15.
        # If you only treat [rot, vel, pos] and keep bias fixed, it might be 9 x 9.
        # We'll do 15 x 15 to handle bias updates fully.
        self.cov = np.zeros((15, 15), dtype=np.float64)

        # Jacobians of the integrated measurements wrt. bias
        # (also 15x3 or 15x3 each, but typically we focus on partial blocks)
        # We'll keep them in cov for simplicity. 
        # Some references keep separate "Jacobian wrt ba" and "Jacobian wrt bg".

        self.dt_sum = 0.0  # total integrated time

    def reset(self, acc_bias: np.ndarray, gyro_bias: np.ndarray):
        """
        Resets the preintegration with new bias estimates if needed.
        """
        self.acc_bias = acc_bias.copy()
        self.gyro_bias = gyro_bias.copy()

        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.cov = np.zeros((15, 15), dtype=np.float64)
        self.dt_sum = 0.0

    def integrate(self, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: float):
        """
        Integrate a single IMU measurement over time dt.
        acc_meas, gyro_meas: raw measurements (3D each).
        dt: time interval [s]
        """
        # Remove bias
        acc_unbias = acc_meas - self.acc_bias
        gyro_unbias = gyro_meas - self.gyro_bias

        # Current rotation
        Rk = self.delta_R

        # Small rotation from gyro
        dR = so3_exponential(gyro_unbias * dt)
        # Update delta_R
        self.delta_R = Rk @ dR

        # Acc in global frame (approx use Rk)
        acc_global = Rk @ acc_unbias + self.gravity

        # Update delta_p, delta_v (simple mid-point or euler; here euler for demonstration)
        self.delta_p += self.delta_v * dt + 0.5 * acc_global * (dt**2)
        self.delta_v += acc_global * dt

        self.dt_sum += dt

        # ----- Covariance propagation (simplified) -----
        # We'll follow a standard 15x15 error-state approach. 
        # Breaking it down for demonstration:
        # block structure: [ dtheta, dv, dp, dba, dbg ] each 3D => total 15D.
        F = np.eye(15)
        G = np.zeros((15, 12))  # process noise Jacobian

        # dtheta depends on gyro noise => rotation part
        # dtheta' = dtheta + ???  We'll approximate with identity + small dt
        # Actually we do a more detailed approach in Forster, but let's keep it short:

        # Rotation sub-block
        # partial derivative of dtheta wrt dtheta is I
        # partial derivative of dtheta wrt dbg is -I * dt (approx)
        # etc.

        I3 = np.eye(3)
        # Rotation error propagation
        F[0:3, 0:3] = I3  # dtheta -> dtheta
        F[0:3, 12:15] = -I3 * dt  # dtheta wrt dbg

        # Velocity error propagation
        # dv' = dv + R*(acc_unbias)*dt => 
        # partial wrt dtheta: -Rk * skew(acc_unbias)*dt 
        # partial wrt dv: I
        # partial wrt dba: -Rk*dt
        # partial wrt dbg: 0
        acc_skew = skew_symmetric(acc_unbias)
        F[3:6, 0:3] = -Rk @ acc_skew * dt
        F[3:6, 3:6] = I3
        F[3:6, 9:12] = -Rk * dt

        # Position error propagation
        # dp' = dp + dv*dt + 0.5 * Rk*acc_unbias*dt^2
        # partial wrt dp: I
        # partial wrt dv: I*dt
        # partial wrt dtheta: -0.5 * Rk * skew(acc_unbias) * dt^2
        # partial wrt dba: -0.5 * Rk * dt^2
        F[6:9, 0:3] = -0.5 * Rk @ acc_skew * (dt**2)
        F[6:9, 3:6] = I3 * dt
        F[6:9, 6:9] = I3
        F[6:9, 9:12] = -0.5 * Rk * (dt**2)

        # Bias propagation: random walk
        # dba' = dba
        # dbg' = dbg
        F[9:12, 9:12] = I3
        F[12:15, 12:15] = I3

        # Noise propagation G
        # IMU noise: [acc_noise, gyro_noise, acc_rw, gyro_rw] -> 12 dims total
        # For real usage, these should be scaled properly by dt / sqrt(dt).
        # We'll keep a simplified version here:
        G[0:3, 3:6] = -I3 * dt          # gyro noise -> dtheta
        G[3:6, 0:3] = -Rk * dt         # acc noise -> dv
        G[6:9, 0:3] = -0.5 * Rk * dt**2 # acc noise -> dp
        G[9:12, 6:9] = I3 * dt         # acc bias random walk
        G[12:15, 9:12] = I3 * dt       # gyro bias random walk

        # Continuous noise covariance
        # Forster approach: block diag of sigma_acc^2, sigma_gyro^2, sigma_acc_bias^2, sigma_gyro_bias^2
        Q = np.diag([
            self.acc_noise**2, self.acc_noise**2, self.acc_noise**2,
            self.gyro_noise**2, self.gyro_noise**2, self.gyro_noise**2,
            self.acc_rw**2,    self.acc_rw**2,    self.acc_rw**2,
            self.gyro_rw**2,   self.gyro_rw**2,   self.gyro_rw**2
        ])

        # Discrete propagation
        # P' = F * P * F^T + G * Q * G^T
        self.cov = F @ self.cov @ F.T + G @ Q @ G.T

    def get_delta(self):
        """
        Returns the pre-integrated results:
          R, v, p, total_dt
        as well as the covariance if needed.
        """
        return self.delta_R, self.delta_v, self.delta_p, self.dt_sum, self.cov

if __name__ == "__main__":
    """
    Example usage:
    Suppose we have an IMU stream at 200 Hz, and we want to pre-integrate
    from time tk to time tk+1 (which might be a camera frame boundary).

    We'll accumulate IMU measurements in a loop, then finalize get_delta().
    """
    # Example parameters (tune for your sensor):
    acc_bias_init = np.array([0.0, 0.0, 0.0])
    gyro_bias_init = np.array([0.0, 0.0, 0.0])
    acc_noise = 0.02       # m/s^2 / sqrt(Hz)
    gyro_noise = 0.0017    # rad/s / sqrt(Hz)
    acc_rw = 0.0002        # bias random walk
    gyro_rw = 0.0001

    integrator = IMUPreintegrator(acc_bias_init, gyro_bias_init,
                                  acc_noise, gyro_noise,
                                  acc_rw, gyro_rw)

    # Example IMU data (pretend we have 10 samples each 0.005s)
    dt = 0.005
    for i in range(10):
        # For test, let's simulate a constant rotation + small accel
        sim_acc = np.array([0.1, 0.0, 9.81])  # ~ stationary, except x-acc
        sim_gyro = np.array([0.0, 0.0, 0.05]) # small yaw rate
        integrator.integrate(sim_acc, sim_gyro, dt)

    # Now we get the integrated result from t0 to t0+10*dt = 0.05s
    dR, dV, dP, dT, cov = integrator.get_delta()

    print("=== Pre-integration result ===")
    print("delta_R:\n", dR)
    print("delta_v:", dV)
    print("delta_p:", dP)
    print("Sum of dt:", dT)
    print("Covariance shape:", cov.shape)
    print("Cov diag:", np.diag(cov))
