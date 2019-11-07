/* 线性回归(使用eigen库实现) */
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    double alip;
    int i, m = 8;
    MatrixXd X(m, 2), Y(m, 1), th(2, 1);

    X.setOnes(m, 2);
    X.col(1) << 0.5, 1.7, 1.9, 2.4, 3.0, 3.2, 3.5, 3.6;
    Y << 1.9, 2.8, 2.2, 3.6, 3.2, 2.3, 3.0, 3.8;
    th << 0.0, 0.0;
    alip = 0.1;

    for(i = 0; i < 1000; i++)
    {
        double sum0 = 0.0, sum1 = 0.0;
        MatrixXd sub(m, 1);
        sub = (X * th) - Y;
        sum0 = sub.sum();
        sum1 = (sub.array() * X.col(1).array()).sum();
        th(0) -= (sum0 * alip / m);
        th(1) -= (sum1 * alip / m);
    }
    cout << "ans =" << endl << th << endl;
    return 0;
}
