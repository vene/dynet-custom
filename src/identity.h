#pragma once

#include <dynet/dynet.h>
#include <dynet/nodes-macros.h>
#include <dynet/expr.h>

using namespace dynet;

using std::string;
using std::vector;

namespace myproj
{

    struct Identity : public Node {

        explicit Identity(const std::initializer_list<VariableIndex>& a) : Node(a)
        {
            this->has_cuda_implemented = false;
        }
        DYNET_NODE_DEFINE_DEV_IMPL();

    };

    Expression identity(const Expression& x);
}
