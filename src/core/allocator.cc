#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); it++)
        {
            if (it->second >= size)
            {
                size_t addr = it->first;
                size_t addr_size = it->second - size;
                freeBlocks.erase(it);
                if (addr_size > 0)
                {
                    freeBlocks[addr + size] = addr_size;
                }
                this->used += size;
                this->peak = std::max(this->used, this->peak);
                return addr;
            }
        }

        throw std::runtime_error("No enough memory");
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t alignedSize = getAlignedSize(size);

        // 1. 先将当前释放的块加入空闲列表
        freeBlocks[addr] = alignedSize;

        // 2. 尝试合并相邻的空闲块（关键步骤）
        auto it = freeBlocks.find(addr);

        // 检查前一个块是否相邻（前一个块的结束地址 = 当前块的起始地址）
        if (it != freeBlocks.begin())
        {
            auto prevIt = std::prev(it);
            size_t prevEnd = prevIt->first + prevIt->second;
            if (prevEnd == addr)
            {
                // 合并前一个块和当前块
                addr = prevIt->first;
                alignedSize += prevIt->second;
                freeBlocks.erase(prevIt);
                freeBlocks.erase(it);
                freeBlocks[addr] = alignedSize;
                it = freeBlocks.find(addr); // 更新迭代器
            }
        }

        // 检查后一个块是否相邻（当前块的结束地址 = 后一个块的起始地址）
        auto nextIt = std::next(it);
        if (nextIt != freeBlocks.end())
        {
            size_t currentEnd = it->first + it->second;
            if (currentEnd == nextIt->first)
            {
                // 合并当前块和后一个块
                alignedSize += nextIt->second;
                freeBlocks.erase(it);
                freeBlocks.erase(nextIt);
                freeBlocks[addr] = alignedSize;
            }
        }

        // 更新内存使用统计
        used -= alignedSize;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
