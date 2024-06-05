#ifndef _MPC_H_
#define _MPC_H_

#include <Eigen/Dense>

// 模板函数不支持分离编译（在一个编译单元中，若模板函数不做实例化，则连接器无法识别）
template <std::size_t N>
Eigen::VectorXd RunMPC(
    const Eigen::VectorXd &init_x,
    const Eigen::MatrixXd &A,
    const Eigen::MatrixXd &B,
    const std::array<Eigen::MatrixXd, N> &Q,
    const std::array<Eigen::MatrixXd, N> &R,
    const Eigen::MatrixXd &F) {
    unsigned int state_num = init_x.rows();
    unsigned int control_num = B.cols();

    Eigen::MatrixXd C, M, temp;
    C.resize(state_num * (N + 1), control_num * N);
    C.setZero();
    M.resize(state_num * (N + 1), state_num);
    M.block(0, 0, state_num, state_num).setIdentity();
    temp.resize(state_num, state_num);
    temp.setIdentity();

    for (unsigned int i = 1; i <= N; ++i) {
        C.block(state_num * i, 0, state_num, control_num) = temp * B;
        C.block(state_num * i, control_num, state_num, control_num * (N - 1)) =
            C.block(state_num * (i - 1), 0, state_num, control_num * (N - 1));

        temp = A * temp;
        M.block(state_num * i, 0, state_num, state_num) = temp;
    }

    Eigen::MatrixXd Q_bar, R_bar;
    Q_bar.resize(state_num * (N + 1), state_num * (N + 1));
    Q_bar.setIdentity();
    R_bar.resize(control_num * N, control_num * N);
    R_bar.setIdentity();

    Q_bar.block(state_num * N, state_num * N, state_num, state_num) = F;

    for (unsigned int i = 0; i < N; ++i) {
        Q_bar.block(state_num * i, state_num * i, state_num, state_num) = Q[i];
        R_bar.block(control_num * i, control_num * i, control_num, control_num) = R[i];
    }

    Eigen::MatrixXd H = C.transpose() * Q_bar * C + R_bar;
    Eigen::MatrixXd E = C.transpose() * Q_bar * M;

    return H.inverse() * (-E * init_x);
}

#endif