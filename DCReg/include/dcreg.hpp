//
// Created by xchu on 19/6/2025.
//

#ifndef DCREG_DCREG_H
#define DCREG_DCREG_H


// we must use some structure from utils
#include "utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ICPRunner {

//    using namespace ICPRunner;

    class DCReg {
    private:
        Config config_;


    public:
        DCReg() = default;

        ~DCReg() = default;


        // 设置参数
        void setConfig(const Config &params) {
            config_ = params;
        }

        // 在XICPCore类的public部分添加
        const Config &getConfig() const {
            return config_;
        }

        DegeneracyAnalysisResult analyzeDegeneracy(
                const Eigen::Matrix<double, 6, 6> &matAtA,
                DetectionMethod detection_method,
                HandlingMethod handling_method) {

            DegeneracyAnalysisResult result;
            result.degenerate_mask.assign(6, false);
            result.W_adaptive = Eigen::Matrix<double, 6, 6>::Zero();
            result.P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();

            // Reference axes for alignment
            Eigen::Vector3d ref_axes[3];
            ref_axes[0] << 1.0, 0.0, 0.0;
            ref_axes[1] << 0.0, 1.0, 0.0;
            ref_axes[2] << 0.0, 0.0, 1.0;

            // Perform necessary analyses
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_full;
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd_full;

            // Full EVD analysis
            es_full.compute(matAtA);
            if (es_full.info() == Eigen::Success) {
                result.eigenvalues_full = es_full.eigenvalues();
                if (result.eigenvalues_full.size() == 6) {
                    double min_eig_trans = std::max(std::abs(result.eigenvalues_full(0)), 1e-12);
                    double max_eig_trans = std::abs(result.eigenvalues_full(2));
                    result.cond_full_sub_trans = max_eig_trans / min_eig_trans;
                    double min_eig_rot = std::max(std::abs(result.eigenvalues_full(3)), 1e-12);
                    double max_eig_rot = std::abs(result.eigenvalues_full(5));
                    result.cond_full_sub_rot = max_eig_rot / min_eig_rot;
                }
            } else {
                result.cond_full_sub_rot = result.cond_full_sub_trans = std::numeric_limits<double>::infinity();
                result.eigenvalues_full.fill(NAN);
            }

            // Full SVD analysis
            svd_full.compute(matAtA, Eigen::ComputeThinU | Eigen::ComputeThinV);
            result.singular_values = svd_full.singularValues();
            if (result.singular_values.size() == 6 && result.singular_values(5) > 1e-12) {
                result.cond_full = result.singular_values(0) / result.singular_values(5);
            } else {
                result.cond_full = std::numeric_limits<double>::infinity();
            }

            // Determine degeneracy based on detection method
            result.isDegenerate = false;
            result.degenerate_mask.assign(6, false);
            switch (detection_method) {
                // 修正后的 SCHUR_CONDITION_NUMBER 检测逻辑
                case DetectionMethod::SCHUR_CONDITION_NUMBER: {
                    break;
                }

                case DetectionMethod::FULL_EVD_MIN_EIGENVALUE: {
                    if (es_full.info() == Eigen::Success && result.eigenvalues_full.allFinite()) {
                        for (int i = 0; i < 6; ++i) {
                            if (result.eigenvalues_full(i) < config_.icp_params.DEGENERACY_THRES_EIG) {
                                result.isDegenerate = true;
                                result.degenerate_mask[i] = true;
                            }
                        }
                    }
                    break;
                }

                case DetectionMethod::EVD_SUB_CONDITION: {
                    result.isDegenerate = (result.cond_diag_rot > config_.icp_params.DEGENERACY_THRES_COND ||
                                           result.cond_diag_trans > config_.icp_params.DEGENERACY_THRES_COND);
                    if (result.isDegenerate) {
                        // only detect is degenerate exists
                        // need to revise!!!!!!!!!!!!!!!!!!!
                        if (result.cond_diag_trans > config_.icp_params.DEGENERACY_THRES_COND) {
                            for (int i = 0; i < 3; ++i) result.degenerate_mask[i + 3] = true;
                        }
                        if (result.cond_diag_rot > config_.icp_params.DEGENERACY_THRES_COND) {
                            for (int i = 0; i < 3; ++i) result.degenerate_mask[i] = true;
                        }
                    }
                    break;
                }

                case DetectionMethod::FULL_SVD_CONDITION: {
                    result.isDegenerate = (result.cond_full > config_.icp_params.DEGENERACY_THRES_COND);
                    if (result.isDegenerate && result.singular_values.allFinite()) {
                        double max_sv = result.eigenvalues_full.maxCoeff();
//                    std::cout << "max lambada: " << max_sv << std::endl;
                        for (int i = 0; i < 6; ++i) {
                            // double cond = max_sv / result.singular_values(i);
                            double cond = max_sv / result.eigenvalues_full(i);
                            // std::cout << "cond_" << i << " " << cond << std::endl;
                            if (cond > config_.icp_params.DEGENERACY_THRES_COND) {
                                // different order of svd and evd
                                // we directly use evd
                                result.degenerate_mask[i] = true;
                            }
                        }
                    }
                    bool debug = false;
                    if (debug) {
                        std::cout << "FULL_SVD_CONDITION Degenerate Mask: ";
                        for (bool mask : result.degenerate_mask) {
                            std::cout << (mask ? "1" : "0") << " ";
                        }
                        std::cout << "" << std::endl;
                    }
                    break;
                }
                case DetectionMethod::NONE_DETE: {
                    result.isDegenerate = false;
                    break;
                }

                default:
                    result.isDegenerate = false;
                    break;
            }


            return result;
        }

        Eigen::VectorXd solveDegenerateSystem(
                const Eigen::Matrix<double, 6, 6> &matAtA,
                const Eigen::VectorXd &matAtB,
                HandlingMethod handling_method,
                const DegeneracyAnalysisResult &analysis) {

            Eigen::VectorXd matX;
            try {
                switch (handling_method) {
                    case HandlingMethod::STANDARD_REGULARIZATION: {
                        Eigen::Matrix<double, 6, 6> H_reg = matAtA;
                        if (analysis.isDegenerate) {
                            H_reg.diagonal().array() += config_.icp_params.STD_REG_GAMMA;
                        }
                        matX = H_reg.colPivHouseholderQr().solve(matAtB);
                        break;
                    }

                    case HandlingMethod::PRECONDITIONED_CG: {
                        if (analysis.isDegenerate) {
                            // wait to implement
                        } else {
                            matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        }
                        break;
                    }

                    case HandlingMethod::SOLUTION_REMAPPING: {
                        // First solve normally
                        matX = matAtA.colPivHouseholderQr().solve(matAtB);

                        // Then apply projection if degenerate
                        if (analysis.isDegenerate && analysis.eigenvalues_full.allFinite()) {
                            Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> es_full(matAtA);
                            if (es_full.info() == Eigen::Success) {
                                Eigen::Matrix<double, 6, 6> P_projector = Eigen::Matrix<double, 6, 6>::Zero();
                                int good_dims = 0;
                                for (int i = 0; i < 6; ++i) {
                                    if (!analysis.degenerate_mask[i]) {
                                        P_projector +=
                                                es_full.eigenvectors().col(i) *
                                                es_full.eigenvectors().col(i).transpose();
                                        good_dims++;
                                    }
                                }
                                if (good_dims > 0) {
                                    matX = P_projector * matX;
                                } else {
                                    matX.setZero();
                                }
                            }
                        }
                        break;
                    }

                    case HandlingMethod::TRUNCATED_SVD: {
                        Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd_full(matAtA,
                                                                                Eigen::ComputeThinU |
                                                                                Eigen::ComputeThinV);
                        if (svd_full.computeU() && svd_full.computeV() && analysis.singular_values.allFinite()) {
                            Eigen::Matrix<double, 6, 1> Sigma_prime_inv_diag = Eigen::Matrix<double, 6, 1>::Zero();
                            int retained_dims = 0;

                            for (int i = 0; i < 6; ++i) {
                                if (!analysis.degenerate_mask[i] && analysis.singular_values(i) > 1e-9) {
                                    Sigma_prime_inv_diag(i) = 1.0 / analysis.singular_values(i);
                                    retained_dims++;
                                }
                            }

                            if (retained_dims == 0) {
                                matX.setZero();
                            } else {
                                matX = svd_full.matrixV() * Sigma_prime_inv_diag.asDiagonal() *
                                       svd_full.matrixU().transpose() * matAtB;
                            }
                        } else {
                            matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        }
                        break;
                    }

                    case HandlingMethod::NONE_HAND: {
                        matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        break;
                    }
                    default: {
                        matX = matAtA.colPivHouseholderQr().solve(matAtB);
                        break;
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "[Solver Error] Exception: " << e.what() << std::endl;
                matX = Eigen::VectorXd::Zero(6);
            }
            return matX;
        }


        bool alignAndOrthonormalize(const Eigen::Matrix3d &V_raw,       // 原始特征向量 (列)
                                    const Eigen::Vector3d &lambda_raw,   // 原始特征值
                                    const Eigen::Vector3d *refs,         // 参考轴
                                    Eigen::Matrix3d &V_aligned,          // 输出：对齐并正交化后的基
                                    std::vector<int> &original_indices)  // 输出：original_indices[i] 是 V_aligned.col(i) 对应的 V_raw 中的列索引
        {
            // wait to implement
    
            return true;
        }


        Eigen::VectorXd solvePCG(const Eigen::Matrix<double, 6, 6> &A,
                                 const Eigen::VectorXd &b,
                                 const Eigen::Matrix<double, 6, 6> &P,
                                 int max_iterations,
                                 double tolerance) {

            // wait to implement
            return Eigen::VectorXd::Zero(6);
        }

    };

}


#endif //DCREG_DCREG_H
