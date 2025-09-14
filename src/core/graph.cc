#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    bool isInverse(const TransposeObj &self, const TransposeObj &other)
    {
        auto self_permute = self.getPermute();
        auto other_permute = other.getPermute();
        for (int i = 0; i < (int)self_permute.size(); ++i)
        {
            if (self_permute[other_permute[i]] != i)
            {
                return false;
            }
        }
        return true;
    }

    int isTransMat(const TransposeObj &self)
    {
        auto permute = self.getPermute();
        auto rank = permute.size();
        if (rank <= 1)
        {
            return -1;
        }
        for (int i = 0; i < (int)rank - 2; ++i)
        {
            if (permute[i] != i)
            {
                return -1;
            }
        }
        return permute[rank - 1] == (int)rank - 1 ? 0 : 1;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 前置校验：确保计算图拓扑有序（从输入到输出的顺序遍历）
        IT_ASSERT(topo_sort(), "Graph is not topologically sorted, optimize failed!");

        std::vector<Operator> remove_ops;   // 待删除算子
        std::vector<Tensor> remove_tensors; // 待删除张量
        std::vector<Tensor> wait_for_cut;   // 待递归清理的无消费者张量

        // -------------------------- 优化1：移除相邻反向Transpose（从后继出发） --------------------------
        // 遍历所有算子，通过"当前算子的后继"寻找连续Transpose
        for (auto const &curr_op : ops)
        {
            // 若当前算子不是Transpose，跳过（我们关注"Transpose → 后继Transpose"的模式）
            if (curr_op->getOpType() != OpType::Transpose)
            {
                continue;
            }

            // 从当前算子的后继中寻找Transpose（核心：从后继出发）
            for (auto const &succ_op : curr_op->getSuccessors())
            {
                // 若后继不是Transpose，跳过
                if (succ_op->getOpType() != OpType::Transpose)
                {
                    continue;
                }

                // 校验两个Transpose是否互逆（当前算子curr_op → 后继succ_op）
                auto trans_curr = static_cast<TransposeObj *>(curr_op.get());
                auto trans_succ = static_cast<TransposeObj *>(succ_op.get());
                if (!isInverse(*trans_curr, *trans_succ))
                {
                    continue; // 非反向操作，不优化
                }

                // 关键张量：
                // curr_input（当前Transpose的输入）→ curr_output（当前的输出）→ succ_output（后继的输出）
                auto curr_input = curr_op->getInputs()[0];
                auto curr_output = curr_op->getOutputs()[0];
                auto succ_output = succ_op->getOutputs()[0];
                auto succ_successors = succ_op->getSuccessors(); // 后继的后继（最终要接回curr_input）

                // 步骤1：断开当前算子与后继算子的依赖
                curr_op->removeSuccessors(succ_op);
                succ_op->removePredecessors(curr_op);

                // 步骤2：断开后继算子与它的后继的依赖
                for (auto const &nxt_op : succ_successors)
                {
                    succ_op->removeSuccessors(nxt_op);
                    nxt_op->removePredecessors(succ_op);
                }

                // 步骤3：重建依赖（跳过两个Transpose）
                // 让当前Transpose的输入直接对接后继的后继
                for (auto const &nxt_op : succ_successors)
                {
                    // 更新后继的后继的输入：从succ_output改为curr_input
                    nxt_op->replaceInput(succ_output, curr_input);
                    // 更新后继的后继的前驱：从succ_op改为curr_op的前驱
                    if (!curr_op->getPredecessors().empty())
                    {
                        auto curr_prev = curr_op->getPredecessors()[0];
                        nxt_op->addPredecessors(curr_prev);
                        curr_prev->addSuccessors(nxt_op);
                    }
                    // 更新curr_input的消费者：添加后继的后继
                    curr_input->addTarget(nxt_op);
                }

                // 步骤4：清理中间张量的依赖
                curr_output->removeTarget(succ_op); // curr_output不再被succ_op使用
                curr_input->removeTarget(curr_op);  // curr_input不再被curr_op使用

                // 步骤5：标记待删除对象
                remove_ops.push_back(curr_op); // 删除当前Transpose
                remove_ops.push_back(succ_op); // 删除后继Transpose
                remove_tensors.push_back(curr_output);
                remove_tensors.push_back(succ_output);

                // 若中间张量无其他消费者，加入递归清理队列
                if (curr_output->getTargets().empty())
                {
                    wait_for_cut.push_back(curr_output);
                }
                if (succ_output->getTargets().empty())
                {
                    wait_for_cut.push_back(succ_output);
                }
            }
        }

        // -------------------------- 优化2：合并Transpose到MatMul（从后继出发） --------------------------
        for (auto const &curr_op : ops)
        {
            // 若当前算子不是Transpose，跳过（我们关注"Transpose → 后继MatMul"的模式）
            if (curr_op->getOpType() != OpType::Transpose)
            {
                continue;
            }

            // 从当前Transpose的后继中寻找MatMul（核心：从后继出发）
            for (auto const &succ_op : curr_op->getSuccessors())
            {
                if (succ_op->getOpType() != OpType::MatMul)
                {
                    continue; // 后继不是MatMul，跳过
                }
                auto matmul_op = static_cast<MatmulObj *>(succ_op.get());

                // 校验Transpose是否仅交换最后两个维度
                auto trans_obj = static_cast<TransposeObj *>(curr_op.get());
                int is_last_two = isTransMat(*trans_obj);
                if (is_last_two != 1)
                {
                    continue; // 不符合合并条件
                }

                // 关键张量：
                // trans_input（Transpose的输入）→ trans_output（Transpose的输出）→ MatMul的输入
                auto trans_input = curr_op->getInputs()[0];
                auto trans_output = curr_op->getOutputs()[0];

                // 步骤1：判断Transpose的输出是MatMul的哪个输入（A还是B）
                bool is_input_a = (succ_op->getInputs()[0] == trans_output);
                bool is_input_b = (succ_op->getInputs()[1] == trans_output);
                if (!is_input_a && !is_input_b)
                {
                    continue; // 理论上不可能，保险校验
                }

                // 步骤2：断开Transpose与MatMul的依赖
                curr_op->removeSuccessors(succ_op);
                succ_op->removePredecessors(curr_op);
                trans_output->removeTarget(succ_op); // 移除MatMul对trans_output的引用

                // 步骤3：MatMul直接使用Transpose的输入（跳过Transpose）
                succ_op->replaceInput(trans_output, trans_input);
                trans_input->addTarget(succ_op); // trans_input新增消费者MatMul

                // 步骤4：更新MatMul的转置属性（合并Transpose的逻辑）
                if (is_input_a)
                {
                    matmul_op->setTransA(matmul_op->getTransA() ^ true); // 异或实现转置叠加
                }
                else
                {
                    matmul_op->setTransB(matmul_op->getTransB() ^ true);
                }

                // 步骤5：标记待删除对象（若Transpose的输出无其他消费者）
                if (trans_output->getTargets().empty())
                {
                    wait_for_cut.push_back(trans_output);
                }
            }
        }

        // -------------------------- 递归清理无效节点 --------------------------
        while (!wait_for_cut.empty())
        {
            auto tensor = wait_for_cut.back();
            wait_for_cut.pop_back();

            if (auto prod_op = tensor->getSource())
            { // 张量的生产者算子
                // 断开生产者与前驱的连接
                for (auto const &pred : prod_op->getPredecessors())
                {
                    pred->removeSuccessors(prod_op);
                    prod_op->removePredecessors(pred);
                }
                // 断开生产者与输入张量的连接
                for (auto const &input : prod_op->getInputs())
                {
                    input->removeTarget(prod_op);
                    if (input->getTargets().empty())
                    {
                        wait_for_cut.push_back(input);
                    }
                }
                // 标记删除
                remove_ops.push_back(prod_op);
                remove_tensors.push_back(tensor);
            }
        }
        for (auto const &op : remove_ops)
        {
            ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
        }
        for (auto const &tensor : remove_tensors)
        {
            tensors.erase(std::remove(tensors.begin(), tensors.end(), tensor), tensors.end());
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        std::vector<size_t> offsets;
        // Add all the tensors to the allocator
        for (const auto &tensor : tensors)
        {
            offsets.push_back(allocator.alloc(tensor->getBytes()));
        }

        const auto base = reinterpret_cast<char *>(allocator.getPtr());

        for (size_t i = 0; i < tensors.size(); ++i)
        {
            auto tensor = tensors[i];
            tensor->setDataBlob(make_ref<BlobObj>(runtime, base + offsets[i]));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini