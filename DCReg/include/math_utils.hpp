#ifndef SO3_UTILS_HPP
#define SO3_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace MathUtils {

// Skew-symmetric matrix from 3D vector
    inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
        Eigen::Matrix3d S;
        S << 0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;
        return S;
    }

    // Exponential map: so(3) -> SO(3)
    inline Eigen::Matrix3d exp(const Eigen::Vector3d &omega) {
        double theta = omega.norm();
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity() + skew(omega);
        }

        Eigen::Vector3d axis = omega / theta;
        Eigen::Matrix3d K = skew(axis);

        // Rodrigues' formula
        return Eigen::Matrix3d::Identity() +
               std::sin(theta) * K +
               (1 - std::cos(theta)) * K * K;
    }

    // Logarithm map: SO(3) -> so(3)
    inline Eigen::Vector3d log(const Eigen::Matrix3d &R) {
        double trace = R.trace();
        double theta = std::acos((trace - 1) / 2);

        if (std::abs(theta) < 1e-10) {
            return Eigen::Vector3d::Zero();
        }

        Eigen::Matrix3d skew_omega = theta / (2 * std::sin(theta)) * (R - R.transpose());
        return Eigen::Vector3d(skew_omega(2, 1), skew_omega(0, 2), skew_omega(1, 0));
    }

    // 辅助函数：SO(3)指数映射
    inline Eigen::Matrix3d exp_map_so3(const Eigen::Vector3d &omega) {
        double theta = omega.norm();
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d omega_hat;
        omega_hat << 0, -omega(2), omega(1),
                omega(2), 0, -omega(0),
                -omega(1), omega(0), 0;

        return Eigen::Matrix3d::Identity() +
               (sin(theta) / theta) * omega_hat +
               ((1 - cos(theta)) / (theta * theta)) * omega_hat * omega_hat;
    }

    // Right Jacobian of SO(3)
    inline Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d &omega) {
        double theta = omega.norm();
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
        }

        Eigen::Vector3d axis = omega / theta;
        Eigen::Matrix3d K = skew(axis);

        return Eigen::Matrix3d::Identity() -
               ((1 - std::cos(theta)) / theta) * K +
               ((theta - std::sin(theta)) / theta) * K * K;
    }

    // Inverse of right Jacobian
    inline Eigen::Matrix3d rightJacobianInvSO3(const Eigen::Vector3d &omega) {
        double theta = omega.norm();
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity() + 0.5 * skew(omega);
        }

        Eigen::Vector3d axis = omega / theta;
        Eigen::Matrix3d K = skew(axis);

        double half_theta = 0.5 * theta;
        double cot_half = 1.0 / std::tan(half_theta);

        return Eigen::Matrix3d::Identity() - 0.5 * K +
               (1.0 / theta - cot_half) * K * K;
    }


    // Compute Jacobian for point-to-plane ICP in SE(3) parameterization
    // point_body: point in body frame
    // normal: plane normal in world frame
    // Returns: 1x6 Jacobian [∂r/∂ω, ∂r/∂v]
    inline Eigen::Matrix<double, 1, 6> computePointToPlaneJacobian(
            const Eigen::Vector3d &point_body,
            const Eigen::Vector3d &normal,
            const Eigen::Matrix3d &R_current) {

        Eigen::Matrix<double, 1, 6> J;

        // Point in world frame
        Eigen::Vector3d point_world = R_current * point_body;

        // Jacobian w.r.t rotation: n^T * (-R * [p]_×)
        // where [p]_× is skew-symmetric matrix of point_body
        Eigen::Matrix3d point_skew = skew(point_body);
        J.block<1, 3>(0, 0) = -normal.transpose() * R_current * point_skew;

        // Jacobian w.r.t translation: n^T * R
        J.block<1, 3>(0, 3) = normal.transpose() * R_current;

        return J;
    }


    // transform form euler to lie
    inline Eigen::Matrix3d  computeEulerToLieJacobian(double roll, double pitch, double yaw) {
        double cr = cos(roll);
        double sr = sin(roll);
        double cp = cos(pitch);
        double sp = sin(pitch);
        Eigen::Matrix3d J;
        if (std::abs(cp) < 1e-6) { return Eigen::Matrix3d::Identity(); } // Gimbal lock check
        J << 1.0, 0.0, sp,
                0.0, cr, -sr * cp,
                0.0, sr, cr * cp;
        return J.inverse().eval(); // Need mapping from Euler rates to angular velocity
    }


    // State on manifold SE(3): [R, t]
    struct SE3State {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;

        SE3State() : R(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()) {}

        SE3State(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation)
                : R(rotation), t(translation) {}

        Eigen::Matrix4d matrix() const {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = R;
            T.block<3, 1>(0, 3) = t;
            return T;
        }

        // Update state on manifold: new_state = current ⊞ delta
        // delta = [omega, v] where omega ∈ so(3), v ∈ R³
        SE3State boxplus(const Eigen::Matrix<double, 6, 1> &delta) const {
            Eigen::Vector3d omega = delta.head<3>();
            Eigen::Vector3d v = delta.tail<3>();

            SE3State result;
            result.R = R * exp(omega);
            result.t = t + R * v;
            return result;
        }

        // 左乘 boxplus - 用于ICP等场景
        SE3State boxplus_left(const Eigen::Matrix<double, 6, 1> &delta) const {
            Eigen::Vector3d omega = delta.head<3>();
            Eigen::Vector3d v = delta.tail<3>();

            SE3State result;
            Eigen::Matrix3d delta_R = exp(omega);
            result.R = delta_R * R;  // 左乘
            result.t = delta_R * t + v;
            return result;
        }

        // 伴随矩阵 Ad(T)
        Eigen::Matrix<double, 6, 6> Adjoint() const {
            Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();

            // 构建t的反对称矩阵
            Eigen::Matrix3d tx;
            tx << 0, -t(2), t(1),
                    t(2), 0, -t(0),
                    -t(1), t(0), 0;

            // Ad = [R, tx*R; 0, R]
            Ad.block<3, 3>(0, 0) = R;
            Ad.block<3, 3>(0, 3) = tx * R;
            Ad.block<3, 3>(3, 3) = R;

            return Ad;
        };
    };


    // Ceres二次代价函数
    struct QuadraticCostFunctor {
        Eigen::Matrix<double, 6, 6> A;
        Eigen::Matrix<double, 6, 1> b;

        QuadraticCostFunctor(const Eigen::Matrix<double, 6, 6> &A_,
                             const Eigen::Matrix<double, 6, 1> &b_) : A(A_), b(b_) {}

        template<typename T>
        bool operator()(const T *const x, T *residual) const {
            Eigen::Matrix<T, 6, 1> x_vec;
            for (int i = 0; i < 6; ++i) x_vec(i) = x[i];

            // 计算 0.5 * x^T * A * x + b^T * x
            Eigen::Matrix<T, 6, 1> Ax = A.cast<T>() * x_vec;
            T quadratic_term = T(0.5) * x_vec.dot(Ax);
            T linear_term = b.cast<T>().dot(x_vec);

            residual[0] = quadratic_term + linear_term;
            return true;
        }
    };
} // namespace SO3Utils

#endif // SO3_UTILS_HPP