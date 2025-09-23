//
// Created by xchu on 13/6/2025.
//

#ifndef CLOUD_MAP_EVALUATION_HESSIAN_COMPUTER_H
#define CLOUD_MAP_EVALUATION_HESSIAN_COMPUTER_H

// --- 5. 并行化的Hessian计算和退化检测 ---

// 方法1：使用TBB的parallel_reduce进行矩阵乘法
class HessianComputer {
    const Eigen::Matrix<double, Eigen::Dynamic, 6> &matA_;
    const Eigen::VectorXd &matB_;

public:
    Eigen::Matrix<double, 6, 6> matAtA;
    Eigen::Matrix<double, 6, 1> matAtB;

    HessianComputer(const Eigen::Matrix<double, Eigen::Dynamic, 6> &matA,
                    const Eigen::VectorXd &matB)
            : matA_(matA), matB_(matB) {
        matAtA.setZero();
        matAtB.setZero();
    }

    HessianComputer(HessianComputer &other, tbb::split)
            : matA_(other.matA_), matB_(other.matB_) {
        matAtA.setZero();
        matAtB.setZero();
    }

    void operator()(const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
            // 获取第i行
            Eigen::Matrix<double, 1, 6> row_i = matA_.row(i);

            // 贡献到 A^T * A
            matAtA.noalias() += row_i.transpose() * row_i;

            // 贡献到 A^T * b
            matAtB.noalias() += row_i.transpose() * matB_(i);
        }
    }

    void join(const HessianComputer &other) {
        matAtA += other.matAtA;
        matAtB += other.matAtB;
    }
};

// usage: 使用parallel_reduce
//HessianComputer hessian_computer(matA, matB);
//tbb::parallel_reduce(tbb::blocked_range<int>(0, laserCloudSelNum),
//        hessian_computer);
//matAtA = hessian_computer.matAtA;
//matAtB = hessian_computer.matAtB;


// --- 5. 高性能并行Hessian计算（方法3：利用对称性）---

// 由于 A^T*A 是对称矩阵，我们只需要计算上三角部分
class SymmetricHessianComputer {
    const Eigen::Matrix<double, Eigen::Dynamic, 6> &matA_;
    const Eigen::VectorXd &matB_;

public:
    // 使用一维数组存储上三角部分（21个元素）
    std::array<double, 21> upper_triangle;  // 6*(6+1)/2 = 21
    Eigen::Matrix<double, 6, 1> matAtB;

    SymmetricHessianComputer(const Eigen::Matrix<double, Eigen::Dynamic, 6> &matA,
                             const Eigen::VectorXd &matB)
            : matA_(matA), matB_(matB) {
        std::fill(upper_triangle.begin(), upper_triangle.end(), 0.0);
        matAtB.setZero();
    }

    SymmetricHessianComputer(SymmetricHessianComputer &other, tbb::split)
            : matA_(other.matA_), matB_(other.matB_) {
        std::fill(upper_triangle.begin(), upper_triangle.end(), 0.0);
        matAtB.setZero();
    }

    void operator()(const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
            const auto &row = matA_.row(i);

            // 计算上三角部分
            int idx = 0;
            for (int j = 0; j < 6; ++j) {
                for (int k = j; k < 6; ++k) {
                    upper_triangle[idx++] += row(j) * row(k);
                }
            }

            // 计算 A^T * b
            for (int j = 0; j < 6; ++j) {
                matAtB(j) += row(j) * matB_(i);
            }
        }
    }

    void join(const SymmetricHessianComputer &other) {
        for (int i = 0; i < 21; ++i) {
            upper_triangle[i] += other.upper_triangle[i];
        }
        matAtB += other.matAtB;
    }

    // 将上三角形式转换回完整矩阵
    Eigen::Matrix<double, 6, 6> getFullMatrix() const {
        Eigen::Matrix<double, 6, 6> result;
        int idx = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                result(i, j) = upper_triangle[idx];
                result(j, i) = upper_triangle[idx];
                idx++;
            }
        }
        return result;
    }
};

// usage: 使用优化的计算
//SymmetricHessianComputer sym_computer(matA, matB);
//tbb::parallel_reduce(tbb::blocked_range<int>(0, laserCloudSelNum),
//        sym_computer);
//matAtA = sym_computer.getFullMatrix();
//matAtB = sym_computer.matAtB;

#endif //CLOUD_MAP_EVALUATION_HESSIAN_COMPUTER_H
