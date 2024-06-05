#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

#include <Python.h>
#define WITHOUT_NUMPY
#include "matplotlib_cpp/matplotlibcpp.h"

#include "mpc.h"

namespace plt = matplotlibcpp;

int main() {
    Eigen::Matrix<double, 2, 2> A;
    Eigen::Matrix<double, 2, 1> B;
    A << 1, 0.1, 0, 2;
    B << 0, 0.5;

    const std::size_t N = 3;
    const unsigned int horz = static_cast<int>(N);
    unsigned int state_num = A.rows();
    unsigned int control_num = B.cols();

    std::array<Eigen::MatrixXd, horz> Q, R;  // compile-time constant
    for (unsigned int i = 0; i < horz; ++i) {
        Q[i].resize(state_num, state_num);
        Q[i].setIdentity();
        R[i].resize(control_num, control_num);
        R[i] = 0.1 * R[i].setIdentity();
    }
    Eigen::MatrixXd F;
    F.resize(state_num, state_num);
    F << 2, 0, 0, 2;

    Eigen::VectorXd init_x;
    init_x.resize(2, 1);
    init_x << 5.0, 5.0;

    std::vector<double> state_0;
    std::vector<double> time;
    state_0.emplace_back(init_x.x());
    time.emplace_back(0.0);

    for (unsigned int i = 0; i < 200; ++i) {
        std::cout << "error: " << init_x.transpose() << std::endl;
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        Eigen::VectorXd control = RunMPC<N>(init_x, A, B, Q, R, F);
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> used_time = (end_time - start_time);
        std::cout << "Once mpc use time(ms): " << used_time.count() * 1000 << std::endl;
        init_x = A * init_x + B * control.x();
        state_0.emplace_back(init_x.x());
        time.emplace_back(0.1 * (i + 1));
    }
    
    plt::plot(time, state_0, "r-");
    plt::xlim(-0.0, 20.0);
    plt::ylim(-0.0, 7.0);
    plt::title("MPC");
    plt::show();

    return 0;
}