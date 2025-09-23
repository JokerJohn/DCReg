#include "xicp.h"


namespace XICP {

// SingleDimensionConstraint implementation
    SingleDimensionConstraint::SingleDimensionConstraint(int dim) : dimension(dim) {}

    template<typename T>
    bool SingleDimensionConstraint::operator()(const T *const x, T *residual) const {
        residual[0] = x[dimension];
        return true;
    }

// Point2PlaneLinearCostFunctor implementation
    Point2PlaneLinearCostFunctor::Point2PlaneLinearCostFunctor(const Eigen::Matrix<double, 6, 6> &A_,
                                                               const Eigen::Matrix<double, 6, 1> &b_) : A(A_), b(b_) {}

    template<typename T>
    bool Point2PlaneLinearCostFunctor::operator()(const T *const x, T *residual) const {
        // 计算线性系统的残差: r = Ax - b
        // 这样最小化 ||r||^2 等价于求解 Ax = b
        for (int i = 0; i < 6; ++i) {
            residual[i] = T(0);
            for (int j = 0; j < 6; ++j) {
                residual[i] += T(A(i, j)) * x[j];
            }
            residual[i] -= T(b(i));
        }
        return true;
    }

// Point2PlaneResidualAutoDiff implementation
    Point2PlaneResidualAutoDiff::Point2PlaneResidualAutoDiff(const Eigen::Vector3d &src,
                                                             const Eigen::Vector3d &tgt,
                                                             const Eigen::Vector3d &n)
            : src_point(src), tgt_point(tgt), normal(n) {}

    template<typename T>
    bool Point2PlaneResidualAutoDiff::operator()(const T *const delta, T *residual) const {
        // delta = [omega, v] 其中 omega是旋转增量，v是平移增量
        Eigen::Matrix<T, 3, 1> omega;
        Eigen::Matrix<T, 3, 1> v;
        for (int i = 0; i < 3; ++i) {
            omega(i) = delta[i];
            v(i) = delta[i + 3];
        }

        // 应用增量变换（注意这里是左乘形式）
        // p_new = p + omega × p + v
        Eigen::Matrix<T, 3, 1> src_T;
        for (int i = 0; i < 3; ++i) {
            src_T(i) = T(src_point(i));
        }

        // 计算 omega × p （叉积）
        Eigen::Matrix<T, 3, 1> omega_cross_p;
        omega_cross_p(0) = omega(1) * src_T(2) - omega(2) * src_T(1);
        omega_cross_p(1) = omega(2) * src_T(0) - omega(0) * src_T(2);
        omega_cross_p(2) = omega(0) * src_T(1) - omega(1) * src_T(0);

        // 变换后的点
        Eigen::Matrix<T, 3, 1> transformed_point = src_T + omega_cross_p + v;

        // 点到平面距离
        T distance = T(0);
        for (int i = 0; i < 3; ++i) {
            distance += (transformed_point(i) - T(tgt_point(i))) * T(normal(i));
        }

        residual[0] = distance;
        return true;
    }

// Point2PlaneResidualNumeric implementation
    Point2PlaneResidualNumeric::Point2PlaneResidualNumeric(const Eigen::Vector3d &src,
                                                           const Eigen::Vector3d &tgt,
                                                           const Eigen::Vector3d &n)
            : src_point(src), tgt_point(tgt), normal(n) {}

    bool Point2PlaneResidualNumeric::operator()(const double *const delta, double *residual) const {
        // 构建变换矩阵（小角度近似）
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R(0, 1) = -delta[2];
        R(0, 2) = delta[1];
        R(1, 0) = delta[2];
        R(1, 2) = -delta[0];
        R(2, 0) = -delta[1];
        R(2, 1) = delta[0];

        Eigen::Vector3d t(delta[3], delta[4], delta[5]);

        // 变换点
        Eigen::Vector3d transformed = R * src_point + t;

        // 计算点到平面距离
        residual[0] = normal.dot(transformed - tgt_point);
        return true;
    }

// DirectionConstraint implementation
    DirectionConstraint::DirectionConstraint(const Eigen::VectorXd &direction, double target_value)
            : direction_(direction), target_value_(target_value) {}

    template<typename T>
    bool DirectionConstraint::operator()(const T *const x, T *residual) const {
        *residual = T(0.0);
        for (int i = 0; i < direction_.size(); ++i) {
            *residual += x[i] * T(direction_(i));
        }
        *residual -= T(target_value_);
        return true;
    }

// InequalityDirectionConstraint implementation
    InequalityDirectionConstraint::InequalityDirectionConstraint(const Eigen::VectorXd &direction, double bound)
            : direction_(direction), bound_(bound) {}

    template<typename T>
    bool InequalityDirectionConstraint::operator()(const T *const x, T *residual) const {
        T projection = T(0.0);
        for (int i = 0; i < direction_.size(); ++i) {
            projection += x[i] * T(direction_(i));
        }

        // 不等式约束：如果投影在界限内，残差为0
        // 否则，残差是超出界限的部分
        T abs_projection = ceres::abs(projection);
        if (abs_projection <= T(bound_)) {
            *residual = T(0.0);
        } else {
            *residual = abs_projection - T(bound_);
        }
        return true;
    }

// XICPCore implementation
    template<typename T>
    void XICPCore<T>::setParameters(const DegeneracyDetectionParameters<T> &params) {
        params_ = params;
    }

    template<typename T>
    const DegeneracyDetectionParameters<T> &XICPCore<T>::getParameters() const {
        return params_;
    }

    template<typename T>
    bool XICPCore<T>::detectDegeneracy(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 更新点数
        params_.numberOfPoints = sourcePoints.cols();

        // 根据方法选择检测策略
        switch (params_.degeneracyAwarenessMethod) {
            case DegeneracyAwarenessMethod::kOptimizedEqualityConstraints:
                return detectLocalizabilityOptimized(sourcePoints, targetNormals, hessian, results);
            case DegeneracyAwarenessMethod::kEqualityConstraints:
            case DegeneracyAwarenessMethod::kInequalityConstraints:
                return detectLocalizabilityTernary(sourcePoints, targetPoints, targetNormals, hessian, results);
            case DegeneracyAwarenessMethod::kSolutionRemapping:
                return detectLocalizabilitySolutionRemapping(hessian, results);

            default:
                return false;
        }
    }

    template<typename T>
    typename XICPCore<T>::Matrix6
    XICPCore<T>::getDegenerateDirections(const LocalizabilityAnalysisResults<T> &results) const {
        Matrix6 directions = Matrix6::Zero();

        // 检查旋转方向
        for (int i = 0; i < 3; ++i) {
            if (results.localizabilityRpy_(i) == static_cast<T>(LocalizabilityCategory::kNonLocalizable)) {
                directions.col(i).head(3) = results.rotationEigenvectors_.col(i);
            }
        }

        // 检查平移方向
        for (int i = 0; i < 3; ++i) {
            if (results.localizabilityXyz_(i) == static_cast<T>(LocalizabilityCategory::kNonLocalizable)) {
                directions.col(i + 3).tail(3) = results.translationEigenvectors_.col(i);
            }
        }

        return directions;
    }

    template<typename T>
    typename XICPCore<T>::Vector6
    XICPCore<T>::getConstraintValues(const LocalizabilityAnalysisResults<T> &results) const {
        Vector6 constraintValues;
        constraintValues.head(3) = results.localizabilityConstraints_.rotationConstraintValues_;
        constraintValues.tail(3) = results.localizabilityConstraints_.translationConstraintValues_;
        return constraintValues;
    }

    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeres(
            const Eigen::Matrix<double, 6, 6> &hessian,
            const Eigen::Matrix<double, 6, 1> &b,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution) {

        // 初始化解
        double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // 创建Ceres问题
        ceres::Problem problem;

        // 使用正确的线性系统代价函数
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<Point2PlaneLinearCostFunctor, 6, 6>(
                        new Point2PlaneLinearCostFunctor(hessian, b)), nullptr, x);


        // 获取退化方向和约束值
        Eigen::Matrix<double, 6, 6> degenerateDirections = getDegenerateDirections(xicpResults);
        Eigen::Matrix<double, 6, 1> constraintValues = getConstraintValues(xicpResults);
        // 判断是等式约束还是不等式约束
        bool is_inequality = (getParameters().degeneracyAwarenessMethod ==
                              XICP::DegeneracyAwarenessMethod::kInequalityConstraints);
        double inequalityBoundMultiplier = getParameters().inequalityBoundMultiplier;

        // std::cout << "inequalityBoundMultiplier: " << inequalityBoundMultiplier << std::endl;

        // 正确的逻辑：只处理退化方向
        int num_constraints = 0;

        // 处理旋转方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);

            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                Eigen::VectorXd direction = degenerateDirections.col(i);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                }
            }
        }

        // 处理平移方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i + 3);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                double constraint_val = constraintValues(i + 3);  // 平移约束值在后3个
                Eigen::VectorXd direction = degenerateDirections.col(i + 3);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;

                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                }
            }
        }

        // 设置Ceres求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 1;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;

        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 复制结果
        for (int i = 0; i < 6; ++i) {
            solution(i) = x[i];
        }

        if (!summary.IsSolutionUsable()) {
            std::cerr << "[XICP-Ceres] Solver failed: " << summary.message << std::endl;
            std::cout << "[XICP-Ceres] Falling back to standard SVD solution..." << std::endl;

            // 如果Ceres失败，使用标准方法求解
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double singular_threshold = 1e-6;
            Eigen::Matrix<double, 6, 1> singular_values = svd.singularValues();
            Eigen::Matrix<double, 6, 1> inv_singular_values = Eigen::Matrix<double, 6, 1>::Zero();

            for (int i = 0; i < 6; ++i) {
                if (singular_values(i) > singular_threshold) {
                    inv_singular_values(i) = 1.0 / singular_values(i);
                }
            }

            solution = svd.matrixV() * inv_singular_values.asDiagonal() * svd.matrixU().transpose() * b;
            std::cout << "[XICP-Ceres] Fallback solution: " << solution.transpose() << std::endl;
        }
    }

    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeresAutoDiff(
            const std::vector <Eigen::Vector3d> &valid_src,
            const std::vector <Eigen::Vector3d> &valid_tgt,
            const std::vector <Eigen::Vector3d> &valid_normals,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution,
            bool useNumericDiff) {

        // 初始化解
        double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // 创建Ceres问题
        ceres::Problem problem;

        // bool useNumericDiff = false;
        if (useNumericDiff) {
            // 使用数值微分而不是自动微分
            for (size_t i = 0; i < valid_src.size(); ++i) {
                ceres::CostFunction *cost_function =
                        new ceres::NumericDiffCostFunction<Point2PlaneResidualNumeric,
                                ceres::CENTRAL, 1, 6>(
                                new Point2PlaneResidualNumeric(valid_src[i], valid_tgt[i], valid_normals[i]));
                problem.AddResidualBlock(cost_function, nullptr, x);
            }
        } else {
            // 添加点到平面的残差块
            for (size_t i = 0; i < valid_src.size(); ++i) {
                ceres::CostFunction *cost_function =
                        new ceres::AutoDiffCostFunction<Point2PlaneResidualAutoDiff, 1, 6>(
                                new Point2PlaneResidualAutoDiff(valid_src[i], valid_tgt[i], valid_normals[i]));
                problem.AddResidualBlock(cost_function, nullptr, x);
            }
        }

        // 获取退化方向和约束值
        Eigen::Matrix<double, 6, 6> degenerateDirections = getDegenerateDirections(xicpResults);
        Eigen::Matrix<double, 6, 1> constraintValues = getConstraintValues(xicpResults);

        // 判断是等式约束还是不等式约束
        bool is_inequality = (getParameters().degeneracyAwarenessMethod ==
                              XICP::DegeneracyAwarenessMethod::kInequalityConstraints);
        double inequalityBoundMultiplier = getParameters().inequalityBoundMultiplier;

        int num_constraints = 0;
        // 处理旋转方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);

            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                Eigen::VectorXd direction = degenerateDirections.col(i);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Rotation axis " << i << std::endl;
                }
            }
        }

        // 处理平移方向
        for (int i = 0; i < 3; ++i) {
            double constraint_val = constraintValues(i + 3);
            double weight = inequalityBoundMultiplier * (1.0 - constraint_val);
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {

                double constraint_val = constraintValues(i + 3);  // 平移约束值在后3个
                Eigen::VectorXd direction = degenerateDirections.col(i + 3);

                if (is_inequality) {
                    // 不等式约束：使用计算出的约束值作为界限
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::InequalityDirectionConstraint, 1, 6>(
                                    new XICP::InequalityDirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;
                    std::cout << "[XICP-Ceres] Added inequality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                } else {
                    // 等式约束：约束值应该是0（或者使用计算出的值）
                    // 根据ICP.cpp，等式约束使用计算出的约束值
                    ceres::CostFunction *constraint_function =
                            new ceres::AutoDiffCostFunction<XICP::DirectionConstraint, 1, 6>(
                                    new XICP::DirectionConstraint(direction, constraint_val));
                    problem.AddResidualBlock(constraint_function,
                                             new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP), x);
                    num_constraints++;

                    std::cout << "[XICP-Ceres] Added equality constraint with bound: " << constraint_val
                              << ", weight: " << weight << ", in Translation axis " << i << std::endl;
                }
            }
        }

        // 设置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 1;  // 只进行一次迭代（相当于一次高斯牛顿步）
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;

        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 复制结果
        for (int i = 0; i < 6; ++i) {
            solution(i) = x[i];
        }
    }

    template<typename T>
    void XICPCore<T>::solveDegenerateSystemWithCeresKKT(
            const Eigen::Matrix<double, 6, 6> &hessian,
            const Eigen::Matrix<double, 6, 1> &b,
            const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
            Eigen::Matrix<double, 6, 1> &solution) {

        // 收集退化的方向索引（模仿原始实现的readLocalizabilityFlags）
        std::vector<int> degenerateIndices;

        // 先检查旋转子空间
        for (int i = 0; i < 3; ++i) {
            if (xicpResults.localizabilityRpy_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                degenerateIndices.push_back(i);
            }
        }

        // 再检查平移子空间
        for (int i = 0; i < 3; ++i) {
            if (xicpResults.localizabilityXyz_(i) ==
                static_cast<double>(XICP::LocalizabilityCategory::kNonLocalizable)) {
                degenerateIndices.push_back(i + 3);
            }
        }

        const int numberOfConstraints = degenerateIndices.size();

        if (numberOfConstraints == 0) {
            // 没有约束，直接求解
            std::cout << "[XICP-KKT] No constraints, solving unconstrained system" << std::endl;
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double tolerance = 1e-6;
            Eigen::Matrix<double, 6, 1> singularValues = svd.singularValues();
            Eigen::Matrix<double, 6, 1> invSingularValues = Eigen::Matrix<double, 6, 1>::Zero();

            for (int i = 0; i < 6; ++i) {
                if (singularValues(i) > tolerance) {
                    invSingularValues(i) = 1.0 / singularValues(i);
                }
            }

            solution = svd.matrixV() * invSingularValues.asDiagonal() * svd.matrixU().transpose() * b;
            return;
        }

        std::cout << "[XICP-KKT] Solving with " << numberOfConstraints << " constraints" << std::endl;

        // 构建增广系统 - 这是关键修正
        // 增广矩阵的结构：[A C^T; C 0]
        Eigen::MatrixXd augmentedA = Eigen::MatrixXd::Zero(6 + numberOfConstraints,
                                                           6 + numberOfConstraints);
        Eigen::VectorXd augmentedb = Eigen::VectorXd::Zero(6 + numberOfConstraints);

        // 填充原始系统 A 和 b
        augmentedA.topLeftCorner(6, 6) = hessian;
        augmentedb.head(6) = b;

        // 构建约束矩阵 C 和约束向量 c
        // 这是最重要的修正部分
        int constraintCounter = 0;
        const int translationIndexOffset = 3;

        for (const auto &index : degenerateIndices) {
            const bool inRotationSubspace = (index < translationIndexOffset);

            // 构建6维约束向量
            Eigen::VectorXd constraintVector = Eigen::VectorXd::Zero(6);

            if (inRotationSubspace) {
                // 旋转约束：约束向量的前3个分量是对应的特征向量
                constraintVector.head(3) = xicpResults.rotationEigenvectors_.col(index);

                // 根据ICP.cpp，等式约束的约束值通常是0
                // 但如果计算出了特定值，也可以使用
                augmentedb(6 + constraintCounter) =
                        xicpResults.localizabilityConstraints_.rotationConstraintValues_(index);
            } else {
                // 平移约束：约束向量的后3个分量是对应的特征向量
                int transIndex = index - translationIndexOffset;
                constraintVector.tail(3) = xicpResults.translationEigenvectors_.col(transIndex);

                augmentedb(6 + constraintCounter) =
                        xicpResults.localizabilityConstraints_.translationConstraintValues_(transIndex);
            }

            // 填充约束矩阵（对称填充）
            // C矩阵的第constraintCounter行
            augmentedA.block(6 + constraintCounter, 0, 1, 6) = constraintVector.transpose();
            // C^T矩阵的第constraintCounter列
            augmentedA.block(0, 6 + constraintCounter, 6, 1) = constraintVector;

            constraintCounter++;
        }

        // 调试输出
        if (getParameters().isPrintingEnabled) {
            std::cout << "[XICP-KKT] Augmented system size: " << augmentedA.rows() << "x" << augmentedA.cols()
                      << std::endl;
            std::cout << "[XICP-KKT] Constraint values: ";
            for (int i = 0; i < numberOfConstraints; ++i) {
                std::cout << augmentedb(6 + i) << " ";
            }
            std::cout << std::endl;
        }

        // 求解增广系统
        Eigen::VectorXd augmentedSolution(6 + numberOfConstraints);

        // 首先尝试QR分解（遵循ICP.cpp的做法）
        Eigen::HouseholderQR <Eigen::MatrixXd> qr(augmentedA);
        augmentedSolution = qr.solve(augmentedb);

        // 检查求解质量
        double residualNorm = (augmentedA * augmentedSolution - augmentedb).norm();

        if (residualNorm > 1e-3 || augmentedSolution.hasNaN()) {
            std::cout << "[XICP-KKT] QR decomposition unstable (residual: " << residualNorm
                      << "), using SVD" << std::endl;

            // 使用SVD作为备选方案
            Eigen::JacobiSVD <Eigen::MatrixXd> svd(augmentedA,
                                                   Eigen::ComputeFullU | Eigen::ComputeFullV);

            // 使用相对容差
            double tolerance = 1e-10 * svd.singularValues()(0);

            Eigen::VectorXd singularValues = svd.singularValues();
            Eigen::VectorXd invSingularValues = Eigen::VectorXd::Zero(singularValues.size());

            int rank = 0;
            for (int i = 0; i < singularValues.size(); ++i) {
                if (singularValues(i) > tolerance) {
                    invSingularValues(i) = 1.0 / singularValues(i);
                    rank++;
                }
            }

            std::cout << "[XICP-KKT] System rank: " << rank << "/" << augmentedA.rows() << std::endl;

            augmentedSolution = svd.matrixV() * invSingularValues.asDiagonal() *
                                svd.matrixU().transpose() * augmentedb;
        }

        // 提取前6个分量作为解
        solution = augmentedSolution.head(6);

        // 输出拉格朗日乘子（用于调试）
        if (numberOfConstraints > 0) {
            Eigen::VectorXd lambda = augmentedSolution.tail(numberOfConstraints);
            std::cout << "[XICP-KKT] Lagrange multipliers: " << lambda.transpose() << std::endl;
        }

        // 验证约束满足情况
        double constraintResidual = 0.0;
        for (int i = 0; i < numberOfConstraints; ++i) {
            double violation = 0.0;
            int index = degenerateIndices[i];

            if (index < 3) {
                // 旋转约束
                Eigen::Vector3d constraint_dir = xicpResults.rotationEigenvectors_.col(index);
                violation = constraint_dir.dot(solution.head(3)) -
                            xicpResults.localizabilityConstraints_.rotationConstraintValues_(index);
            } else {
                // 平移约束
                int transIndex = index - 3;
                Eigen::Vector3d constraint_dir = xicpResults.translationEigenvectors_.col(transIndex);
                violation = constraint_dir.dot(solution.tail(3)) -
                            xicpResults.localizabilityConstraints_.translationConstraintValues_(transIndex);
            }

            constraintResidual += violation * violation;
            std::cout << "[XICP-KKT] Constraint " << i << " violation: " << violation << std::endl;
        }
        constraintResidual = std::sqrt(constraintResidual);
        std::cout << "[XICP-KKT] Total constraint residual: " << constraintResidual << std::endl;

        // 验证解的有效性
        if (solution.hasNaN()) {
            std::cerr << "[XICP-KKT] Invalid solution detected, falling back to unconstrained solution"
                      << std::endl;

            // 回退到无约束解
            Eigen::JacobiSVD <Eigen::Matrix<double, 6, 6>> svd(hessian,
                                                               Eigen::ComputeFullU | Eigen::ComputeFullV);
            double tolerance = 1e-6;
            Eigen::Matrix<double, 6, 1> singularValues = svd.singularValues();
            Eigen::Matrix<double, 6, 1> invSingularValues = Eigen::Matrix<double, 6, 1>::Zero();

            for (int j = 0; j < 6; ++j) {
                if (singularValues(j) > tolerance) {
                    invSingularValues(j) = 1.0 / singularValues(j);
                }
            }

            solution = svd.matrixV() * invSingularValues.asDiagonal() *
                       svd.matrixU().transpose() * b;
        }

        std::cout << "[XICP-KKT] Final solution: " << solution.transpose() << std::endl;
    }

    template<typename T>
    void XICPCore<T>::eigenAnalysis3x3(const Matrix6 &hessian, LocalizabilityAnalysisResults<T> &results) {
        // 旋转部分特征分析
        Eigen::JacobiSVD <Matrix3> svd_rot(hessian.template topLeftCorner<3, 3>(),
                                           Eigen::ComputeFullU | Eigen::ComputeFullV);
        results.rotationEigenvectors_ = svd_rot.matrixU();

        // 平移部分特征分析
        Eigen::JacobiSVD <Matrix3> svd_trans(hessian.template bottomRightCorner<3, 3>(),
                                             Eigen::ComputeFullU | Eigen::ComputeFullV);
        results.translationEigenvectors_ = svd_trans.matrixU();
    }

    template<typename T>
    bool XICPCore<T>::detectLocalizabilityOptimized(
            const Matrix &sourcePoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 3x3特征分析
        eigenAnalysis3x3(hessian, results);

        size_t numPoints = sourcePoints.cols();

        // 计算交叉积（用于旋转对齐）
        Matrix crosses(3, numPoints);
        for (size_t i = 0; i < numPoints; ++i) {
            Vector3 src_pt = sourcePoints.col(i).template head<3>();
            Vector3 tgt_normal = targetNormals.col(i).template head<3>();
            Vector3 cross = src_pt.cross(tgt_normal);
            T norm = cross.norm();
            crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
        }

        // 检测每个特征向量的局部化性
        for (int i = 0; i < 3; ++i) {
            // 检测旋转方向
            T rotContribution = 0.0;
            T rotHighContribution = 0.0;
            bool rotLocalizable = detectDirectionLocalizability(
                    results.rotationEigenvectors_.col(i),
                    crosses,
                    rotContribution,
                    rotHighContribution);

            results.localizabilityRpy_(i) = rotLocalizable ?
                                            static_cast<T>(LocalizabilityCategory::kLocalizable) :
                                            static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            results.localizabilityConstraints_.rotationConstraintValues_(i) = rotLocalizable ? 1.0 : 0.0;

            if (params_.isPrintingEnabled && !rotLocalizable) {
                std::cout << "Rotation axis " << i << " is non-localizable. "
                          << "Combined: " << rotContribution << "/" << params_.enoughInformationThreshold
                          << ", High: " << rotHighContribution << "/" << params_.insufficientInformationThreshold
                          << std::endl;
            }

            // 检测平移方向
            T transContribution = 0.0;
            T transHighContribution = 0.0;
            bool transLocalizable = detectDirectionLocalizability(
                    results.translationEigenvectors_.col(i),
                    targetNormals,
                    transContribution,
                    transHighContribution);

            results.localizabilityXyz_(i) = transLocalizable ?
                                            static_cast<T>(LocalizabilityCategory::kLocalizable) :
                                            static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            results.localizabilityConstraints_.translationConstraintValues_(i) = transLocalizable ? 1.0 : 0.0;

            if (params_.isPrintingEnabled && !transLocalizable) {
                std::cout << "Translation axis " << i << " is non-localizable. "
                          << "Combined: " << transContribution << "/" << params_.enoughInformationThreshold
                          << ", High: " << transHighContribution << "/" << params_.insufficientInformationThreshold
                          << std::endl;
            }
        }

        // 旋转特征向量回到map坐标系
        Matrix3 rot_to_map = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
        for (int i = 0; i < 3; ++i) {
            results.rotationEigenvectors_.col(i) = rot_to_map * results.rotationEigenvectors_.col(i);
            results.translationEigenvectors_.col(i) = rot_to_map * results.translationEigenvectors_.col(i);
        }

        return true;
    }

    template<typename T>
    bool XICPCore<T>::detectLocalizabilityTernary(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 首先进行3x3特征分析
        eigenAnalysis3x3(hessian, results);

        // 计算点云中心
        Vector3 center = Vector3::Zero();
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            center += sourcePoints.col(i).template head<3>();
        }
        center /= static_cast<T>(sourcePoints.cols());

        // 计算交叉积用于旋转对齐
        Matrix crosses(3, sourcePoints.cols());
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            Vector3 src_pt = sourcePoints.col(i).template head<3>() - center;
            Vector3 tgt_normal = targetNormals.col(i).template head<3>();
            Vector3 cross = src_pt.cross(tgt_normal);
            T norm = cross.norm();
            crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
        }

        // 计算deltas（源点和目标点之间的差）
        Matrix deltas(sourcePoints.rows(), sourcePoints.cols());
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            deltas.col(i) = sourcePoints.col(i) - targetPoints.col(i);
        }

        // 创建对齐向量容器
        std::vector <std::pair<Eigen::Index, T>> alignmentList;
        alignmentList.reserve(sourcePoints.cols());

        // 为每个特征向量进行三元级别检测
        for (int eigIdx = 0; eigIdx < 3; ++eigIdx) {
            // 旋转子空间检测
            detectSubspaceLocalizabilityTernary(
                    sourcePoints, targetPoints, targetNormals, crosses, deltas,
                    results.rotationEigenvectors_.col(eigIdx),
                    alignmentList, eigIdx, true, results);

//                 std::cout << "highlyContributingPoints size trans: " << params_.highlyContributingNumberOfPoints  << std::endl;
            params_.highlyContributingNumberOfPoints_rot = params_.highlyContributingNumberOfPoints;

            // 平移子空间检测
            detectSubspaceLocalizabilityTernary(
                    sourcePoints, targetPoints, targetNormals, targetNormals, deltas,
                    results.translationEigenvectors_.col(eigIdx),
                    alignmentList, eigIdx, false, results);
            params_.highlyContributingNumberOfPoints_trans = params_.highlyContributingNumberOfPoints;

//                  std::cout << "highlyContributingPoints size rot: " << params_.highlyContributingNumberOfPoints  << std::endl;
        }

        // 将特征向量旋转回到优化坐标系
        if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
            Matrix3 rot_to_opt = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
            for (int i = 0; i < 3; ++i) {
                results.rotationEigenvectors_.col(i) = rot_to_opt * results.rotationEigenvectors_.col(i);
                results.translationEigenvectors_.col(i) = rot_to_opt * results.translationEigenvectors_.col(i);
            }
        }

        if (params_.isPrintingEnabled) {
            //std::cout << "[XICPCore-Ternary] Detection completed" << std::endl;
            std::cout << "Translation localizability: " << results.localizabilityXyz_.transpose() << std::endl;
            std::cout << "Rotation localizability: " << results.localizabilityRpy_.transpose() << std::endl;
//                std::cout << "highlyContributingPoints size: " << params_.highlyContributingNumberOfPoints << std::endl;
            std::cout << "highlyContributingPoints size: " << params_.highlyContributingNumberOfPoints_trans << " "
                      << params_.highlyContributingNumberOfPoints_rot << std::endl;
        }

        return true;
    }

    template<typename T>
    bool XICPCore<T>::detectLocalizabilitySolutionRemapping(
            const Matrix6 &hessian,
            LocalizabilityAnalysisResults<T> &results) {

        // 6x6特征分析
        Eigen::JacobiSVD <Matrix6> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vector6 eigenvalues = svd.singularValues();
        Matrix6 eigenvectors = svd.matrixU();

        // 计算条件数
        T conditionNumber = eigenvalues(0) / eigenvalues(5);

        if (params_.isPrintingEnabled) {
            std::cout << "[Solution Remapping] Condition number: " << conditionNumber
                      << ", threshold: " << params_.solutionRemappingThreshold << std::endl;
            std::cout << "Eigenvalues: " << eigenvalues.transpose() << std::endl;
        }

        // 构建投影矩阵 - 遵循ICP.cpp的实现
        results.solutionRemappingProjectionMatrix_ = Matrix6::Zero();

        // 根据特征值阈值决定保留哪些方向
        T eigenValueThreshold = params_.solutionRemappingThreshold;

        for (int i = 0; i < 6; ++i) {
            if (eigenvalues(i) >= eigenValueThreshold) {
                // 保留这个方向
                results.solutionRemappingProjectionMatrix_ +=
                        eigenvectors.col(i) * eigenvectors.col(i).transpose();
            } else {
                // 这是一个退化方向 - 更新局部化状态
                if (i < 3) {
                    // 对应旋转的某个轴
                    // 需要将6D特征向量映射到3D子空间
                    Vector3 rot_component = eigenvectors.col(i).head(3);
                    T rot_norm = rot_component.norm();
                    if (rot_norm > 0.5) {  // 主要是旋转退化
                        // 找到最大分量对应的轴
                        int max_idx = 0;
                        T max_val = std::abs(rot_component(0));
                        for (int j = 1; j < 3; ++j) {
                            if (std::abs(rot_component(j)) > max_val) {
                                max_val = std::abs(rot_component(j));
                                max_idx = j;
                            }
                        }
                        results.localizabilityRpy_(max_idx) =
                                static_cast<T>(LocalizabilityCategory::kNonLocalizable);
                    }
                } else {
                    // 对应平移的某个轴
                    Vector3 trans_component = eigenvectors.col(i).tail(3);
                    T trans_norm = trans_component.norm();
                    if (trans_norm > 0.5) {  // 主要是平移退化
                        // 找到最大分量对应的轴
                        int max_idx = 0;
                        T max_val = std::abs(trans_component(0));
                        for (int j = 1; j < 3; ++j) {
                            if (std::abs(trans_component(j)) > max_val) {
                                max_val = std::abs(trans_component(j));
                                max_idx = j;
                            }
                        }
                        results.localizabilityXyz_(max_idx) =
                                static_cast<T>(LocalizabilityCategory::kNonLocalizable);
                    }
                }
            }
        }

        // 如果投影矩阵是零矩阵，设为单位矩阵（避免完全失败）
        if (results.solutionRemappingProjectionMatrix_.norm() < 1e-6) {
            results.solutionRemappingProjectionMatrix_ = Matrix6::Identity();
            if (params_.isPrintingEnabled) {
                std::cout << "[Solution Remapping] Warning: Projection matrix was zero, using identity"
                          << std::endl;
            }
        }

        return true;
    }

    template<typename T>
    bool XICPCore<T>::detectDirectionLocalizability(
            const Vector3 &eigenvector,
            const Matrix &alignmentVectors,
            T &combinedContribution,
            T &highContribution) {

        combinedContribution = 0.0;
        highContribution = 0.0;
        bool informationIsEnough = false;

        for (Eigen::Index i = 0; i < alignmentVectors.cols() && !informationIsEnough; ++i) {
            Vector3 alignVec = alignmentVectors.col(i).template head<3>();
            T alignment = std::abs(alignVec.dot(eigenvector));

            if (alignment >= params_.point2NormalMinimalAlignmentCosineThreshold) {
                combinedContribution += alignment;
            }

            if (alignment >= params_.point2NormalStrongAlignmentCosineThreshold) {
                highContribution += alignment;
            }

            // 检查是否有足够信息（遵循ICP.cpp的逻辑）
            informationIsEnough = (combinedContribution >= params_.enoughInformationThreshold) ||
                                  (highContribution >= params_.insufficientInformationThreshold);
        }

        return informationIsEnough;
    }

    template<typename T>
    bool XICPCore<T>::compareAlignmentList(const std::pair <Eigen::Index, T> &p1,
                                           const std::pair <Eigen::Index, T> &p2) {
        return p1.second > p2.second;
    }

    template<typename T>
    void XICPCore<T>::detectSubspaceLocalizabilityTernary(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix &alignmentVectors,
            const Matrix &deltas,
            const Vector3 &eigenvector,
            std::vector <std::pair<Eigen::Index, T>> &alignmentList,
            int index,
            bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        // 清空对齐列表和重置计数器
        alignmentList.clear();
        params_.contributingNumberOfPoints = 0;
        params_.highlyContributingNumberOfPoints = 0;
        params_.combinedContribution = 0.0;
        params_.highContribution = 0.0;

        // 计算每个点的对齐贡献（使用对齐值排序，遵循ICP.cpp）
        for (Eigen::Index i = 0; i < sourcePoints.cols(); ++i) {
            Vector3 alignVec = alignmentVectors.col(i).template head<3>();
            T alignment = std::abs(alignVec.dot(eigenvector));

            // 存储索引和对齐值（不是贡献值）
            alignmentList.push_back(std::make_pair(i, alignment));

            if (alignment >= params_.point2NormalMinimalAlignmentCosineThreshold) {
                params_.combinedContribution += alignment;
                params_.contributingNumberOfPoints++;
            }

            if (alignment >= params_.point2NormalStrongAlignmentCosineThreshold) {
                params_.highContribution += alignment;
                params_.highlyContributingNumberOfPoints++;
            }
        }

        // 决定局部化级别
        LocalizabilitySamplingType samplingType = decideLocalizabilityLevelTernary(
                index, isRotationSubspace, results);

        // 确定采样点数
        Eigen::Index pointsToSample = 0;
        switch (samplingType) {
            case LocalizabilitySamplingType::kHighContributionPoints:
                pointsToSample = params_.highlyContributingNumberOfPoints;
                break;
            case LocalizabilitySamplingType::kMixedContributionPoints:
                pointsToSample = params_.contributingNumberOfPoints;
                break;
            case LocalizabilitySamplingType::kUnnecessary:
            case LocalizabilitySamplingType::kInsufficientPoints:
                return;  // 不需要采样
        }

        if (params_.isPrintingEnabled) {
//                std::cout << "highlyContributingPoints size: " << params_.highlyContributingNumberOfPoints_trans << " "
//                          << params_.highlyContributingNumberOfPoints_rot << std::endl;
        }

        // 如果需要采样，执行采样和约束计算
        if (pointsToSample > 0) {
            // 确保采样数量合理
            pointsToSample = std::min(pointsToSample, sourcePoints.cols());
            pointsToSample = std::max(pointsToSample,
                                      static_cast<Eigen::Index>(params_.insufficientInformationThreshold));

            // 部分排序获取贡献最大的点
            std::partial_sort(alignmentList.begin(),
                              alignmentList.begin() + pointsToSample,
                              alignmentList.end(),
                              compareAlignmentList);

            // 执行约束求解
            solvePartialConstraints(sourcePoints, targetPoints, targetNormals, deltas,
                                    alignmentList, pointsToSample, eigenvector, index,
                                    isRotationSubspace, results);

//                std::cout << "constriant: " << results.localizabilityConstraints_.transpose() << std::endl;
        }
    }

    template<typename T>
    LocalizabilitySamplingType XICPCore<T>::decideLocalizabilityLevelTernary(
            int index, bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        auto &localizability = isRotationSubspace ?
                               results.localizabilityRpy_(index) : results.localizabilityXyz_(index);
        auto &constraintValue = isRotationSubspace ?
                                results.localizabilityConstraints_.rotationConstraintValues_(index) :
                                results.localizabilityConstraints_.translationConstraintValues_(index);

        // 完全可定位
        if (params_.combinedContribution >= params_.highInformationThreshold ||
            params_.highContribution >= params_.enoughInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kLocalizable);
            constraintValue = 1.0;
            return LocalizabilitySamplingType::kUnnecessary;
        }

        // 部分可定位 - 混合贡献
        if (params_.combinedContribution >= params_.enoughInformationThreshold &&
            params_.combinedContribution < params_.highInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            // 关键区别：等式约束 vs 不等式约束
            if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
                // 等式约束：二值化为0（退化方向）
                constraintValue = 0.0;
            } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
                // 不等式约束：连续值
                constraintValue = params_.inequalityBoundMultiplier *
                                  (params_.combinedContribution / params_.highInformationThreshold);
                constraintValue = std::min(constraintValue, T(1.0));
            }
            return LocalizabilitySamplingType::kMixedContributionPoints;
        }

        // 部分可定位 - 高贡献点
        if (params_.highContribution >= params_.insufficientInformationThreshold) {
            localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);

            if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
                // 等式约束：二值化为0
                constraintValue = 0.0;
            } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
                // 不等式约束：部分约束
                constraintValue = params_.inequalityBoundMultiplier * 0.5;
                constraintValue = std::min(constraintValue, T(1.0));
            }
            return LocalizabilitySamplingType::kHighContributionPoints;
        }

        // 不可定位
        localizability = static_cast<T>(LocalizabilityCategory::kNonLocalizable);
        constraintValue = 0.0;  // 对所有方法都是0
        return LocalizabilitySamplingType::kInsufficientPoints;
    }

    template<typename T>
    void XICPCore<T>::solvePartialConstraints(
            const Matrix &sourcePoints,
            const Matrix &targetPoints,
            const Matrix &targetNormals,
            const Matrix &deltas,
            const std::vector <std::pair<Eigen::Index, T>> &alignmentList,
            Eigen::Index pointsToSample,
            const Vector3 &eigenvector,
            int index,
            bool isRotationSubspace,
            LocalizabilityAnalysisResults<T> &results) {

        // 创建采样点云和相应的法向量、增量
        Matrix sampledPoints(sourcePoints.rows(), pointsToSample);
        Matrix sampledNormals(targetNormals.rows(), pointsToSample);
        Matrix sampledDeltas(deltas.rows(), pointsToSample);

        for (Eigen::Index i = 0; i < pointsToSample; ++i) {
            Eigen::Index idx = alignmentList[i].first;
            sampledPoints.col(i) = sourcePoints.col(idx);
            sampledNormals.col(i) = targetNormals.col(idx);
            sampledDeltas.col(i) = deltas.col(idx);
        }

        // 计算约束值
        T constraintValue = 0.0;
        const T thr = T(1e-5);

        if (!isRotationSubspace) {
            // 平移约束
            Matrix3 partial_A = sampledNormals.topRows(3) * sampledNormals.topRows(3).transpose();

            // 计算点积 dot = dot(deltas, normals)
            Matrix dotProd = Matrix::Zero(1, sampledNormals.cols());
            for (Eigen::Index i = 0; i < sampledNormals.rows(); ++i) {
                dotProd += (sampledDeltas.row(i).array() * sampledNormals.row(i).array()).matrix();
            }

            Vector3 partial_b = -(sampledNormals.topRows(3) * dotProd.transpose());

            // 求解
            Vector3 x_partial;
            if (partial_A.determinant() > thr) {
                x_partial = partial_A.ldlt().solve(partial_b);
            } else {
                // 使用SVD进行更稳定的求解
                Eigen::JacobiSVD <Matrix3> svd(partial_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                x_partial = svd.solve(partial_b);
            }

            // 将特征向量旋转到优化坐标系
            Vector3 rotatedEigenVector = eigenvector;
            if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
                Matrix3 T_op = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
                rotatedEigenVector = T_op * eigenvector;
            }

            constraintValue = rotatedEigenVector.transpose() * x_partial;
        } else {
            // 旋转约束
            // 重新计算采样点的交叉积
            Matrix crosses(3, pointsToSample);
            Vector3 center = sampledPoints.topRows(3).rowwise().mean();

            for (Eigen::Index i = 0; i < pointsToSample; ++i) {
                Vector3 pt = sampledPoints.col(i).head(3) - center;
                Vector3 normal = sampledNormals.col(i).head(3);
                Vector3 cross = pt.cross(normal);
                T norm = cross.norm();
                crosses.col(i) = (norm < 1.0) ? cross : cross.normalized();
            }

            Matrix3 partial_A = crosses * crosses.transpose();

            // 计算点积
            Matrix dotProd = Matrix::Zero(1, sampledNormals.cols());
            for (Eigen::Index i = 0; i < sampledNormals.rows(); ++i) {
                dotProd += (sampledDeltas.row(i).array() * sampledNormals.row(i).array()).matrix();
            }

            Vector3 partial_b = -(crosses * dotProd.transpose());

            // 使用更稳定的求解方法（遵循ICP.cpp的实现）
            Vector3 x_partial;
            Vector3 y;
            if (partial_A.determinant() > thr) {
                // LU分解
                Eigen::PartialPivLU <Matrix3> lu(partial_A);
                Matrix3 l = Matrix3::Identity();
                l.template triangularView<Eigen::StrictlyLower>() = lu.matrixLU();
                Matrix3 u = lu.matrixLU().template triangularView<Eigen::Upper>();

                Matrix3 new_A = l.transpose() * l;
                Vector3 new_b = l.transpose() * (lu.permutationP() * partial_b);

                // 使用double精度进行SVD求解
                y = new_A.template cast<double>().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(
                        new_b.template cast<double>()).
                        template cast<T>();
                x_partial = u.inverse() * y;
            } else {
                x_partial = Vector3::Zero();
            }

            // 旋转特征向量
            Vector3 rotatedEigenVector = eigenvector;
            if (params_.transformationToOptimizationFrame != Eigen::Matrix4d::Identity()) {
                Matrix3 T_op = params_.transformationToOptimizationFrame.template topLeftCorner<3, 3>();
                rotatedEigenVector = T_op * eigenvector;
            }

            constraintValue = rotatedEigenVector.transpose() * x_partial;
        }

        // 更新约束值
        auto &finalConstraintValue = isRotationSubspace ?
                                     results.localizabilityConstraints_.rotationConstraintValues_(index) :
                                     results.localizabilityConstraints_.translationConstraintValues_(index);

        // 对于等式约束，约束值已经在decideLocalizabilityLevelTernary中设置
        // 只有当需要采样求解时才会调用这个函数，此时约束值应该保持为0
        if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kEqualityConstraints) {
            // 等式约束：保持二值（已在decide函数中设置为0）
            // 不需要更新，因为退化方向的约束值应该是0
            if (params_.isPrintingEnabled) {
                std::cout << "[Partial Constraints] "
                          << (isRotationSubspace ? "Rotation" : "Translation")
                          << " axis " << index
                          << " constraint value: " << finalConstraintValue
                          << " (Equality constraint)" << std::endl;
            }
        } else if (params_.degeneracyAwarenessMethod == DegeneracyAwarenessMethod::kInequalityConstraints) {
            // 不等式约束：使用计算出的值并缩放
            finalConstraintValue = std::min(std::abs(constraintValue) * params_.inequalityBoundMultiplier, T(1.0));
            if (params_.isPrintingEnabled) {
                std::cout << "[Partial Constraints] "
                          << (isRotationSubspace ? "Rotation" : "Translation")
                          << " axis " << index
                          << " constraint value: " << finalConstraintValue
                          << " (Inequality constraint)" << std::endl;
            }
        } else {
            // 对于等式约束，保留原始约束值
            finalConstraintValue = constraintValue;
        }
    }

// 模板函数的显式实例化，用于自动微分
    template bool SingleDimensionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool SingleDimensionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool Point2PlaneLinearCostFunctor::operator()<double>(const double *const x, double *residual) const;

    template bool Point2PlaneLinearCostFunctor::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool Point2PlaneResidualAutoDiff::operator()<double>(const double *const delta, double *residual) const;

    template bool Point2PlaneResidualAutoDiff::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const delta, ceres::Jet<double, 6>
    *residual) const;

    template bool DirectionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool DirectionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

    template bool InequalityDirectionConstraint::operator()<double>(const double *const x, double *residual) const;

    template bool InequalityDirectionConstraint::operator()<ceres::Jet < double, 6>>
    (
    const ceres::Jet<double, 6> *const x, ceres::Jet<double, 6>
    *residual) const;

// 显式实例化XICPCore类模板
//    template class XICPCore<float>;
    template
    class XICPCore<double>;

} // namespace XICP