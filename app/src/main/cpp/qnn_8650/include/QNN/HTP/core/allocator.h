//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef ALLOCATOR_H
#define ALLOCATOR_H 1

#include <cstddef>
#include <algorithm>
#include <memory>
#include "dtype_enum.h"
#include "weak_linkage.h"
#include "macros_attribute.h"

enum class MemoryClass {
    Plain,
    TCM,
    UnCached, // for spill/fill DDR
    XXX_LAST_MEMORY_TYPE,
    Default = Plain
};

PUSH_VISIBILITY(default)

class Graph;
namespace fa {
struct PoolDesc;
struct BigBuff;
struct RuntimeAllocator;
} // namespace fa
namespace hnnx {

class Serializer;
class Deserializer;

// some options flags (powers of 2) for calls to Tensor::allocate
enum AllocOptions {
    uncached_int8 = 0x1, // override MemoryClass to UnCached.
    uncached_int16 = 0x2,
    uncached_fp16 = 0x4
};

/*
 * Maybe FIXME: It seems like FancyAllocator has just about all the same interfaces as Allocator,
 * is all this pimpl stuff needed, or could we just inherit Allocator and have a unique_ptr<Allocator>
 * in our graph?
 */

class Allocator {
  public:
    // MIN_ALIGN, MAX_ALIGN:
    //  - both must be powers of 2
    //  -  8 <= MIN_ALIGN <= MAX_ALIGN
    // All allocations will be aligned to at least MIN_ALIGN, both start and end of each region.
    // This includes sub-allocations in memory pools.
    // Alignment requests > MAX_ALIGN may be treated as MAX_ALIGN if allocated in DDR.
    //
    static constexpr unsigned MIN_ALIGN = 256;
    static constexpr unsigned MAX_ALIGN = 256;

    // The alignment used by TCM allocation; >= MIN_ALIGN
    static constexpr unsigned TCM_ALLOC_ALIGN = 2048;

    static void *vacant() { return (void *)2; } // special value for 'vacant' slot.
    enum Mode { AllocVirtual, AllocPhysical, AllocTemp, AllocTempEnd, AllocComplete, LastMode = AllocComplete };

    // AllocTemp/AllocTempEnd are used in Virtual mode, to set a 'Temp Physical' mode
    // where allocation is done to physical memory, but into memory blocks which
    // are discarded when we return via AllocTempEnd (So, AllocTempEnd is not possible as an actual
    // current mode).
    // This is intended to support nesting (multiple levels of AllocTemp; each
    // AllocTempEnd discards all allocs since the matching AllocTemp; but
    // currently nesting is not supported, so AllocTemp must be followed by AllocTempEnd,
    // which actually takes you back to AllocVirtual
    // AllocComplete allows no further allocations. A deserialized allocator
    // is in this state.

    Allocator(Mode mode_in, Graph &graph_in) : graph(graph_in), mode(mode_in){};
    API_EXPORT virtual ~Allocator() = 0;

    Graph &graph;

    // Either allocates enough, or dips into a buffer (and changes the buffer pointer and size parameter accordingly).
    // al is an alignment parameter; it must be a power of 2 or the code below won't work.
    API_EXPORT void *tracked_aligned_alloc(size_t al, size_t bytes, fa::BigBuff *const bb = nullptr);
    API_EXPORT void tracked_free(void *aligned_ptr) noexcept;

    API_EXPORT virtual void allocate_n(void **arrp, size_t n, size_t block_size, size_t alignment, MemoryClass memclass,
                                       unsigned options, DType dtype);

    // options for allocate_persistent_blocks.
    // if 'allnew' is *not* present, it is assumed that all of the pointers
    //   are either null, or point to existing persistent blocks. The 'null' ones
    //   are replaced with new allocations, and the ref counts are increased in both cases.
    // with 'allnew': pointers are assumed to contain garbage. Equivalent to zeroing the
    //   pointer table first.
    //
    // zoneB: with this, ref counts are update in 'B' zone instead of A
    //
    // incref: ovverides 'allnew'; all of the existing pointers are required to be valid persistent
    //    blocks; the ref counts are increased by 1
    // decref: overrides 'incref and allnew'; all of the pointers are required to be valid persistent
    //    blocks; the ref counts are reduced by 1. If total refs are zero, block is freed.
    //    the pointer table is not updated.
    //
    // infinite: newly alloc'd blocks get refcount set to a huge number, instead of 1.
    // Currently this is used when deserializing, since we can't free things immediately when in Crate.
    //
    enum persistent_options {
        allnew = 1, // assume existing pointers are garbage, allocate them all.
        zoneB = 2, // reference count in zone B instead of A.
        incref = 4, // enforce that all existing are persistnent; incref them.
        decref = 8,
        infinite = 16, // refcounts on new blocks, set to a huge # instead of 1.
    };

    // allocate n 'persistent' blocks of the given size/alignment, and update the table.
    API_EXPORT virtual void allocate_persistent_blocks(void **table, size_t nblocks, size_t block_size,
                                                       size_t alignment, unsigned options);

    inline void *allocate(const void *oldval, size_t block_size, size_t alignment, MemoryClass memclass,
                          unsigned options, DType dtype)
    {
        void *tmp = (void *)oldval;
        allocate_n(&tmp, 1, block_size, alignment, memclass, options, dtype);
        return tmp;
    }

    Mode get_mode() const { return mode; }
    API_EXPORT virtual void set_mode(Mode new_mode);

    virtual void set_tcm_pool(void *base, size_t size) = 0;

    virtual void set_largest_memory_alloc_size(size_t size) = 0;

    /*
	 * Serialize all the internal data for the allocator.
	 * Memory regions / pools, etc.
	 */
    API_EXPORT virtual void serialize(Serializer &) const;
    /*
	 * Deserialize the allocator, restore internal data from buffer.
	 * Actually you maybe just want the deserializing constructor....
	 */
    virtual void deserialize(Deserializer &, unsigned char *const weight_data = nullptr,
                             const size_t weight_length = 0) = 0;
    /*
	 * Given a pointer to a block of data, and its size serialize the memory.
	 * This might be serialized as "this is a persistent memory allocation and here is all the persistent data value"
	 * Or it might be serialized as "This is something in a certain pool of memory, here's the offset."
	 */
    API_EXPORT virtual void serialize_block(Serializer &, const void *addr, const size_t block_size);
    /*
	 * Given a serialized representation of data, turn it back into a pointer.
	 */
    virtual void *deserialize_block(Deserializer &) = 0;

    // virtual methods to serialize/deserialize n blocks at once. Allocatoe subclasses do not need to define
    // these, the default mathods just call the _block methods in a loop. But they will reduce code
    // in the templated tensor methods.
    // Tensors should use these multi-block methods consistently in serialize and deserialize.
    // this allows the allocator to use a different strategy from the 'blocks' case.

    API_EXPORT virtual void serialize_blocks(Serializer &, const void *const *table, const size_t block_size,
                                             size_t nblocks);
    API_EXPORT virtual void deserialize_blocks(Deserializer &, const void **table, size_t nblocks);

    static inline constexpr size_t fixup_alignment(size_t align)
    {
        static_assert(MIN_ALIGN >= 8 && (MIN_ALIGN & (MIN_ALIGN - 1)) == 0, "bad MIN_ALIGN");
        static_assert(MAX_ALIGN >= MIN_ALIGN && (MAX_ALIGN & (MAX_ALIGN - 1)) == 0, "bad MAX_ALIGN");
        if (MIN_ALIGN < MAX_ALIGN) {
            return std::max<size_t>(MIN_ALIGN, std::min<size_t>(MAX_ALIGN, align));
        } else {
            return MIN_ALIGN;
        }
    }

    static inline constexpr size_t round_up_align(size_t n, size_t align) { return (n + (align - 1)) & ~(align - 1); }
    template <typename T> static inline T *round_up_align(T *p, size_t align)
    {
        return (T *)round_up_align((size_t)p, align);
    }

  protected:
    Mode mode = AllocVirtual;
};

//
// this is s 'shim' class to help in making dummy allocators. It defines overrides
// for all of the pure-virtual methods, so you don't need to
//
class FakeAllocator : public Allocator {
  public:
    FakeAllocator(Allocator::Mode mode_in, Graph &graph_in) : Allocator(mode_in, graph_in){};
    API_EXPORT virtual ~FakeAllocator();

    API_EXPORT virtual void set_tcm_pool(void *base, size_t size) override;
    API_EXPORT virtual void set_largest_memory_alloc_size(size_t size) override;
    API_EXPORT virtual void deserialize(Deserializer &, unsigned char *weight_data = nullptr,
                                        size_t weight_length = 0) override;
    API_EXPORT virtual void *deserialize_block(Deserializer &) override;
};

// this is an accessor which is used by the Dma 'Fill' operation
// to get a source pointer for reading const, based on (pool_id, offset).
// It also holds the base pointer for ddr spill area.
// Maybe other things could be added later.

class MemPoolRunTimeAccessor {
    void *spill_area;
    fa::PoolDesc const *pool_table; // pool_table[0] is for poolid=1
    unsigned max_pool_id;

  public:
    MemPoolRunTimeAccessor(void *spill_area_in, fa::PoolDesc const *pt, unsigned pt_size)
        : spill_area(spill_area_in), pool_table(pt), max_pool_id(pt_size)
    {
    }
    MemPoolRunTimeAccessor() : spill_area(nullptr), pool_table(nullptr), max_pool_id(0) {}
    MemPoolRunTimeAccessor(MemPoolRunTimeAccessor const &) = default;
    MemPoolRunTimeAccessor &operator=(MemPoolRunTimeAccessor const &) = default;

    // pool ids are >= 1, <= num_pools
    constexpr unsigned num_pools() const { return max_pool_id; }
    // map pool_id to base address of the data, for persistent pool.
    // implementation in fa_alloc.h
    char const *get_persistent_pool_base(unsigned pool_id) const;
    bool get_is_weights(unsigned pool_id) const;
    void *get_spill_area() const { return spill_area; }

    // used to construct the ConstExtentDescriptor during prep
    // implementation in fa_alloc.h
    fa::PoolDesc const *get_descriptor(unsigned pool_id) const;
};

} // namespace hnnx

POP_VISIBILITY()

#endif
