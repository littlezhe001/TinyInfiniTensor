#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shape_A = inputs[0]->getDims();
        auto shape_B = inputs[1]->getDims();
        auto A_rank = shape_A.size();
        auto B_rank = shape_B.size();
        assert(A_rank >= 2 && B_rank >= 2);
        // 是否进行转置操作
        if (transA)
        {
            std::swap(shape_A[A_rank - 1], shape_A[A_rank - 2]);
        }
        if (transB)
        {
            std::swap(shape_B[B_rank - 1], shape_B[B_rank - 2]);
        }
        // 令矩阵乘m×c与c×n的结果维数最后为m×n
        // 使得c为1即可，然后取A,B矩阵对应位置的最大值
        shape_A[A_rank - 1] = 1;
        shape_B[B_rank - 2] = 1;
        auto output_shape = infer_broadcast(shape_A, shape_B);
        return {{output_shape}};
    }

} // namespace infini