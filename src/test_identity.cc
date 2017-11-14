#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <iostream>

#include "identity.h"

namespace dy = dynet;

int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    const unsigned DIM = 5;

    ComputationGraph cg;
    dy::ParameterCollection m;
    auto x_par = m.add_parameters({DIM}, 0, "x");
    Expression x = dy::parameter(cg, x_par);
    std::cout << x.value() << std::endl;
    std::cout << std::endl;

    Expression y = myproj::identity(x);
    cg.forward(y);
    std::cout << y.value() << std::endl;

    for(int k = 0; k < DIM; ++k)
    {
        std::cout << "wrt dimension " << k << std::endl;
        Expression yk = dy::pick(y, k);
        check_grad(m, yk, 2);
    }
}

