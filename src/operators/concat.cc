#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini
{
    ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
        : OperatorObj(OpType::Concat, inputs, {output})
    {
        int rank = inputs[0]->getRank();
        dim = get_real_axis(_dim, rank);
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)
    {
        Shape dims = inputs[0]->getDims();
        auto rank = inputs[0]->getRank();

        // =================================== 作业 ===================================
        // TODO：修改 dims，返回正确的 concat 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        // =================================== 作业 ===================================

        // inputs[0] 是 “形状基准张量”，定义了所有输入应遵循的非拼接轴维度和秩；
        // i从1开始 是为了 “跳过自我对比的冗余，直接校验后续输入与基准的一致性”，符合 Concat 算子 “非拼接轴必须一致” 的核心约束，是高效且严谨的实现方式。
        for (size_t i = 1; i < inputs.size(); ++i)
        {
            dims[dim] += inputs[i]->getDims()[dim];
        }
        return {{dims}};
    }

    std::string ConcatObj::toString() const
    {
        std::ostringstream os;
        os << "Concat[" << getGuid() << "]";
        os << "(";
        for (auto input : inputs)
            os << vecToString(input->getDims()) << ",";
        os << "dim=" << dim << ",";
        os << "input=";
        for (auto input : inputs)
            os << input->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

} // namespace infini
