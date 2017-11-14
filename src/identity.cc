#include "identity.h"

#include <dynet/nodes-macros.h>
#include <dynet/expr.h>

using namespace dynet;
using std::string;
using std::vector;

namespace myproj
{
    string Identity::as_string(const vector<string>& arg_names) const
    {
        std::ostringstream s;
        s << "identity(" << arg_names[0] << ")";
        return s.str();
    }

    Dim Identity::dim_forward(const vector<Dim> &xs) const
    {
        DYNET_ARG_CHECK(xs.size() == 1, "Identity takes a single input");
        return xs[0];
    }

    template<class MyDevice>
    void Identity::forward_dev_impl(const MyDevice& dev,
                                    const vector<const Tensor*>& xs,
                                    Tensor& fx) const
    {
        fx.tvec() = xs[0]->tvec();
    }


    template <class MyDevice>
    void Identity::backward_dev_impl(const MyDevice& dev,
                                     const vector<const Tensor*>& xs,
                                     const Tensor& fx,
                                     const Tensor& dEdf,
                                     unsigned i,
                                     Tensor& dEdxi) const
    {
        (*dEdxi) += (*dEdf);
    }

    DYNET_NODE_INST_DEV_IMPL(Identity)

    Expression identity(const Expression& x) {
        return Expression(x.pg, x.pg->add_function<Identity>({x.i}));
    }
}
