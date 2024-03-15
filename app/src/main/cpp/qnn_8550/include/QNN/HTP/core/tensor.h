//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef HEXNN_TENSOR_H
#define HEXNN_TENSOR_H 1
/*
 * This file is trying to figure out a nice Tensor class, which allows for access
 * to a data structure with potentially unknown underlying data types and layout.
 *
 * What is a Tensor? It's a multidimensional array of data
 * It has a "Rank": the number of dimensions.
 * It has a shape.
 * It contains data values.
 * There is a mechanism to access the data values.
 *
 * From an abstract perspective, that's about all we should have to know about a tensor.
 * However, to form a concrete tensor, it should also be observed that:
 *
 * The data values have some type.  They may be encoded/decoded with some extra information.
 * The data is laid out in some fashion.  It may be a contiguous block, it may have the data
 *   shuffled in some way, it may have a table of pointers to fixed-size blocks...
 * There might be extra padding values around the "true" data
 *
 * To facilitate the most abstract interfaces being available while also being
 * able to specify concrete tensors and have the compiler understand the mechanics
 * of the concrete tensor, we probably need to have:
 * * Abstract tensor as a base class that provides a generic interface always, using runtime polymorphism
 * * Subclasses that provide more concrete tensor representations, finalizing aspects of the tensor,
 * * Concrete classes that provide the compiler with full visibility in the details of the tensor
 *
 * Because values may be encoded/decoded from their internal representation, in the most abstract
 * representation we can not just return a data element.  Instead, we return an accessor object.
 * The accessor object works like an rvalue or lvalue, but is able to decode (rvalue) or encode (lvalue)
 * the data as appropriate.
 * (At least, that's how I think it should work...)
 *
 * We'd like to use the operator() to allow us to have multidimentional-array-indices-like interface,
 * much in the same way we have in Eigen.  So for a 4D tensor, with indices batchidx,row,col,channel,
 * we should be able to say out_tensor(batchidx,row,col,channel) = in_tensor(batchidx,row,col,channel)
 *
 * Although we might consider a variety of different types for tensor internal values, including int32,
 * I propose to use "float" as the defualt interface type.  It should work well for many integers, and
 * is the appropriate interface for real data whether quantized or not.  A reasonable alternative would
 * be double, but double is quite a bit more expensive on Hexagon.  Of course, other methods of accessing
 * the data could be made for different types.
 *
 * Having extremely abstract tensors allows us to have extremely generic
 * functions, but having easily available less abstract tensors should allow us
 * to easily specify constraints for ops that are more optimal or that are
 * demanding certain parameters for their inputs.  For example, if we always want
 * our convolutional op to have a 4D input tensor, we might specify that it is
 * a RankedTensor<4> instead of a Tensor, indicating that the op can only use
 * a tensor with rank 4.
 *
 * Tensors are very fundamental to how we are going to work on things, so it's incredibly important to
 * be on our best behavior here.
 *
 */

/*
 * EJP: FIXME:
 * * The helper classes, Accessors and Interfaces and such, should be in a sub-namespace for cleanliness.
 * * Make the Abstract/Base/Generic naming consistent.  I like "Generic" at the moment.
 */
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>
#include <cstddef>
#include <typeindex>

#include "allocator.h"
#include "shape.h"
#include "serialize_oplist.h"
#include "deserializer.h"
#include "dtype.h"
#include "float16.h"
#include "graph_status.h"
#include "interface_defs.h"
#include "log.h"
#include "memory_layout.h"
#include "padding.h"
#include "template_help.h"
#include "conversions.h"
#include "crate.h"
#include "minihash.h"
#include "macros_attribute.h"
#include "weak_linkage.h"

#define TENSOR_MAGIC 0x1337beef

//#define ALWAYS_INLINE /* NOTHING */

#if 0
/*
 * What is the type of an Index?
 * What is a Signed Index that might be negative?
 * Maybe both of these should just be "int" everywhere.
 */
using Idx = size_t;
using SIdx = long;

#endif

/*
 * This name makes no sense.
 */
#if 0
struct OctetType {
    uint8_t *buf;
    size_t buflen;
};
#endif

#include "weak_linkage.h"
PUSH_VISIBILITY(default)

template <typename T> class PlainInterface;
template <typename T> class ScaleOffsetInterface;

template <> constexpr DType dtype_of_type<ScaleOffsetInterface<uint8_t>>()
{
    return DType::QUInt8;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<int8_t>>()
{
    return DType::QInt8;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<uint16_t>>()
{
    return DType::QUInt16;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<int16_t>>()
{
    return DType::QInt16;
}

template <> constexpr DType dtype_of_type<PlainInterface<Float16>>()
{
    return DType::Float16;
}
template <> constexpr DType dtype_of_type<PlainInterface<float>>()
{
    return DType::Float32;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<NN_INT32_T>>()
{
    return DType::QInt32;
}
template <> constexpr DType dtype_of_type<PlainInterface<NN_INT32_T>>()
{
    return DType::Int32;
}

extern long long int dma_validate_cycles;

namespace hnnx {
extern uint64_t checksum_bytes(uint64_t prev, uint8_t const *bytes, unsigned n);

// this is to solve a circular dependency issue; defined in graph.h
class DMA_Manager;
DMA_Manager *get_dma_manager(Graph const &);
DMA_Manager *get_dma_manager(Deserializer const &);

//
// This type represent a set of block_id, across a tensor or group of
// tensors.
//typedef std::set<void*> blockid_set_t;
// .. but this should work too...
typedef miniset<void *> blockid_set_t;

//
// This is an interface class; a reference to this
// is passed to tensor->enum_memory_blocks; the tensor then calls the 'supply_blocks_func' method
// (maybe using one of the handy wrappers) to generate one or more 'void*' which are the block
// ids.
//   Rules are:
//   - if 'supply_blocks_func' is called with memclass < 0, the memory class of the block is unspecified;
//     if it is called with memclass >=0, the value  is  MemoryClass and the tensor guarantees that all
//     of the blocks are in that class.
//   - Tensor may in general make multiple calls to supply_blocks_func in one call to enum_memory_blocks, and may
//     supply different values of mclass parameter. But currently it is only one.
//   - The tensor does *not* guarantee that the same id is not presented multiple times in one call
//     to enum_memory_blocks.
//
class MemBlockEnumerator {
  public:
    virtual ~MemBlockEnumerator() {}
    virtual void supply_blocks_func(Tensor const *tensp, int memclass, void *const *ptr, size_t num) = 0;
    // Tensors can use these wrappers
    inline void supply_blocks(Tensor const *tensp, void *const *ptr, size_t num)
    {
        supply_blocks_func(tensp, -1, ptr, num);
    }
    inline void supply_blocks(Tensor const *tensp, MemoryClass mc, void *const *ptr, size_t num)
    {
        supply_blocks_func(tensp, int(mc), ptr, num);
    }
};
// utility class, to enumerate to a std::set
// if mclass_sel >=0, we skip tensors which have a different memory class.
class MemBlockEnumToSet : public MemBlockEnumerator {
    blockid_set_t &m_set;
    int m_memclass_sel;

  public:
    explicit MemBlockEnumToSet(blockid_set_t &s, int mclass_sel = -1) : m_set(s), m_memclass_sel(mclass_sel) {}
    MemBlockEnumToSet(blockid_set_t &s, MemoryClass mc) : m_set(s), m_memclass_sel(int(mc)) {}
    virtual void supply_blocks_func(Tensor const *, int memclass, void *const *ptr, size_t num) override
    {
        if (m_memclass_sel >= 0 && memclass >= 0 && m_memclass_sel != memclass) return;
        for (size_t i = 0; i < num; i++) {
            if (ptr[i] != Allocator::vacant()) m_set.emplace(ptr[i]);
        }
    }
};
// This is to support Tensor::enum_memory_blocks_withfunc( ..callable..)
//  and similar for Op methods
template <typename ENFUNC> class MemBlockEnumWrapper : public MemBlockEnumerator {
    ENFUNC m_enfunc;

    virtual void supply_blocks_func(Tensor const *tensp, int memclass, void *const *ptr, size_t num) override
    {
        m_enfunc(tensp, memclass, ptr, num);
    }

  public:
    inline MemBlockEnumWrapper(ENFUNC &&ef) : m_enfunc(std::move(ef)) {}
    inline MemBlockEnumWrapper(ENFUNC const &ef) : m_enfunc(ef) {}
};

// This is to support Tensor::replace_memory_blocks_withfunc( ..callable..)
//  and similar for Op methods
//  The 'replfunc' is called as: void* replfunc( Tensor const *tp, void *old_blkid)
//  for each block in the tensor; the returned value is used as the replacement blkid.
template <typename REPLFUNC> class MemBlockReplBlockWrapper : public MemBlockEnumerator {
    REPLFUNC m_replfunc;

    virtual void supply_blocks_func(Tensor const *tensp, int memclass, void *const *ptr, size_t num) override
    {
        for (unsigned i = 0; i < num; i++) {
            void *newblk = m_replfunc(tensp, ptr[i]);
            const_cast<void *&>(ptr[i]) = newblk;
        }
    }

  public:
    inline MemBlockReplBlockWrapper(REPLFUNC &&ef) : m_replfunc(std::move(ef)) {}
    inline MemBlockReplBlockWrapper(REPLFUNC const &ef) : m_replfunc(ef) {}
};

} // namespace hnnx

/*
 * An Interface has all the necessary values and functionality to encode and decode values
 *
 * virtual methods do generic conversion to/from floats, with a void * to the encoded data.
 *
 * Each concrete Tensor (and some less-than-concrete) has an instance of an Interface.
 *
 * Note: read_floats/write_floats have default implementations in tensor.cc, which
 * use element_size() and read_float/write_float.
 *
 * IMPORTANY: All interface classes must be trivially destructible.
 * As a result, even though we have virtual methods, it is safe to
 * have no virtual dtor.
 * This is important for performance, since every tensor has an Interface subclass
 * embedded in it; and most tensor classes need no destructor for any other reason.
 * So a dtor requirement in the 'interface' could add time to the teardown,
 * even if the dtors don't do very much.
 */

class Interface {
  protected:
    struct qparms {
        int offset;
        float scale;
        float scale_recip;
    };
    API_EXPORT static constexpr qparms null_parms = {0, 1.0f, 1.0f};

  public:
    virtual void write_float(void *ptr, const float in) const noexcept = 0;
    virtual float read_float(const void *ptr) const noexcept = 0;
    virtual size_t element_size() const noexcept = 0;
    API_EXPORT virtual void write_floats(void *ptr, const float *srcp, size_t n) const noexcept;
    API_EXPORT virtual void read_floats(float *dstp, const void *ptr, size_t n) const noexcept;
    virtual bool is_quantized() const noexcept { return false; }

    // these are 'virtual-by-proxy...'
    inline float get_scale() const noexcept { return get_qparms()->scale; }
    inline float get_scale_recip() const noexcept { return get_qparms()->scale_recip; }
    inline int32_t get_offset() const noexcept { return get_qparms()->offset; }
    API_EXPORT bool compare_equal(Interface const &) const noexcept; // implements operator == in general case

  protected:
    virtual qparms const *get_qparms() const noexcept { return &null_parms; }
    virtual bool compare_eq_same_type(Interface const *rhs) const noexcept = 0;
};

namespace hnnx {
// make_interface<INTFC>::from_odef( Graph &, OutputDef const &odef)
// returns a pointer to an INTFC suitable for odef, either
// by finding an existing one, or by adding a new one to the crate.
// make_interface<INTFC>::from_deser(Deseralizer & dctx)
// returns a pointer to an INTFC , deserialized,
// by finding an existing one which matches, or by adding a new one to the crate.
template <typename INTFC> struct make_interface {
    static Interface const *from_odef(Graph &, OutputDef const &odef);
    static Interface const *from_deser(Deserializer &dctx);
};
} // namespace hnnx

/*
 * But guess what... you can't ever instantiate an abstract class!
 * So if we want to return a generic Accessor, we need to make it non-abstract.
 *
 * We need an abstract pointer to some element.  The way to do this is void *
 * We need a pointer to the Interface, which needs to be able to work with a void *
 *
 * This pushes the runtime polymorphism into the Interface, which we can share
 * between Accessor instances
 */

class GenericAccessorRO {
  protected:
    void *data;
    const Interface &interface;

  public:
    GenericAccessorRO(void const *data_in, const Interface &interface_in)
        : data(const_cast<void *>(data_in)), interface(interface_in)
    {
    }
    GenericAccessorRO(GenericAccessorRO const &) = default;
    typedef GenericAccessorRO AccessorRO;
    inline float as_float() const { return interface.read_float(data); }
    inline operator float() const { return as_float(); }
};
class GenericAccessor : public GenericAccessorRO {
  public:
    GenericAccessor(void *data_in, const Interface &interface_in) : GenericAccessorRO(data_in, interface_in) {}
    GenericAccessor(GenericAccessor const &) = default;
    inline void set_float(float v) { interface.write_float(data, v); }
    inline float operator=(float v)
    {
        set_float(v);
        return v;
    }
    inline float operator=(GenericAccessorRO const &rhs)
    {
        float const v = rhs.as_float();
        set_float(v);
        return v;
    }
    inline float operator=(GenericAccessor const &rhs)
    {
        return operator=(static_cast<GenericAccessorRO const &>(rhs));
    }
};
/**
 * For each 'interface' there are a pair of accessor classes
 *   Interface::Accessor
 *   Interface::AccessorRO
 *   .. which the types returned by Tensor(..indices...)
 *
 *  These have the following:
 *       typedef AccessorRO;                            - correponding RO type.
 *       typedef element_type;							- type of the stored element
 *       element_type .value() const;					- direct read
 *       .as_float() const;								- convert to float
 *       operator float() const;						- same
 *  (If not RO):
 *       .set_value( element_type &);                   - direct store
 *       .set_float( float )							- assign from float
 *       operator=( float )								- assign from float
 *       operator=( Accessor const & )				    - assign from same accessor
 *       operator=( AccessorRO const & )				- assign from R/O accessor
 *  The assignment operators may return either a float,
 *    or an AccessorRO by value
 *    or an Accessor const &  (which is *this)
 *    or an AccessorRO const & (only if it's *this by subclass).
 *
 *  Both have copy ctors, and AccessorRO(Accessor const &) works.
 *
 *  AccessorRO may or may not be a direct public base of Accessor
 *
 *  The 'GenericAccessor' and GenericAccessorRO have all of the above, except
 *  for element_type, .value(), and .set_value().
 */

/**
 * @class NullInterface
 *
 * @brief A NullInterface throws away data and returns zero
 */

class NullInterface final : public Interface {
  public:
    inline constexpr NullInterface() {}
    virtual void write_float(void *ptr, const float in) const noexcept override final {}
    virtual float read_float(const void *ptr) const noexcept override final { return 0.0f; }
    virtual size_t element_size() const noexcept override final { return 0; }
    virtual void write_floats(void *ptr, const float *srcp, size_t n) const noexcept override final {}
    API_EXPORT virtual void read_floats(float *dstp, const void *ptr, size_t n) const noexcept override final;
    int compare(const NullInterface &rhs) const { return 0; };
    uint32_t interface_hash() const noexcept { return 0; }

    // hide the slower implementations in the base class...
    inline float get_scale() const noexcept { return 1.0f; }
    inline float get_scale_recip() const noexcept { return 1.0f; }
    inline int32_t get_offset() const noexcept { return 0; }

  private:
    virtual bool compare_eq_same_type(Interface const *rhs) const noexcept override final { return true; }
    // Accessor for NullInterface - empty class.
    struct nullval {
        operator float() const { return 0.0f; }
    };
    class AcsrRO {
      public:
        using element_type = nullval;
        using AccessorRO = AcsrRO;
        AcsrRO() {}
        AcsrRO(void const *, NullInterface const &) {}
        AcsrRO(AcsrRO const &) = default;
        inline element_type value() const { return nullval{}; }
        inline float as_float() const { return 0.0f; }
        inline operator float() const { return 0.0f; }
    };
    class Acsr : public AcsrRO {
      public:
        using element_type = nullval;
        using AccessorRO = AcsrRO;
        Acsr(void *, const NullInterface &) {}
        Acsr(Acsr const &) = default;
        inline void set_float(float v) {}
        inline void set_value(element_type v) {}
        inline float operator=(float v) { return 0.0f; }
        inline float operator=(AcsrRO const &rhs) { return 0.0f; }
        inline float operator=(Acsr const &rhs) { return 0.0f; }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};

// make_interface for NullInterface; easy, just have one
// and return a pointer to it.
template <> struct hnnx::make_interface<NullInterface> {
    API_EXPORT static NullInterface null_ifc; // in tensor.cc
    static Interface const *from_odef(Graph &, OutputDef const &odef) { return &null_ifc; }
    static Interface const *from_deser(Deserializer &dctx) { return &null_ifc; }
};

/**
 * @class PlainInterface
 *
 * @brief A tensor with Floats needs no conversion.
 * You could also use this for integral value tensors where the integral values are the true values;
 * they would get converted to floats.
 */
template <typename T> class PlainInterface final : public Interface {
  public:
    using element_type = T;
    template <typename TX> using interface_other_type = PlainInterface<TX>;
    explicit constexpr PlainInterface(const OutputDef &def) {}
    constexpr PlainInterface() {}
    explicit constexpr PlainInterface(hnnx::Deserializer &) : PlainInterface() {}
    static inline constexpr T convert_from_float(const float &in)
    {
        return saturate_round<T>(in);
    } // except for T=float!
    static inline constexpr float convert_to_float(const T &in) { return float(in); }
    virtual void write_float(void *ptr,
                             const float in) const noexcept override final; // inlined below
    virtual inline float read_float(const void *ptr) const noexcept override final
    {
        auto p = static_cast<const T *>(ptr);
        return convert_to_float(*p);
    }
    virtual size_t element_size() const noexcept override final { return sizeof(T); };
    virtual void write_floats(void *ptr, const float *srcp, size_t n) const noexcept override final;
    virtual void read_floats(float *dstp, const void *ptr, size_t n) const noexcept override final;
    int compare(const PlainInterface &rhs) const { return 0; }
    uint32_t interface_hash() const noexcept { return 1; } // different from NullInterface

    // hide the slower implementations in the base class...
    inline float get_scale() const noexcept { return 1.0f; }
    inline float get_scale_recip() const noexcept { return 1.0f; }
    inline int32_t get_offset() const noexcept { return 0; }

  private:
    virtual bool compare_eq_same_type(Interface const *rhs) const noexcept override final { return true; }
    // Accessor for PlainInterface
    // Doesn't need a reference to interface, just a data pointer (or data, for AcsrRO)
    // We can't actually call it AccessorRO since it needs to contain a typedef AccessorRO.
    //
    class Acsr;
    class AcsrRO {
      protected:
        T val;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        AcsrRO(void const *data_in, PlainInterface const &) : val(*static_cast<T const *>(data_in)) {}
        AcsrRO(AcsrRO const &) = default;
        AcsrRO(Acsr const &a) : val(a.value()) {}
        inline element_type value() const { return val; }
        inline float as_float() const { return convert_to_float(val); }
        inline operator float() const { return as_float(); }
    };
    class Acsr {
      protected:
        T *data;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        Acsr(void *data_in, PlainInterface const &) : data(static_cast<T *>(data_in)) {}
        Acsr(Acsr const &) = default;
        inline element_type value() const { return *data; }
        inline float as_float() const { return convert_to_float(*data); }
        inline operator float() const { return as_float(); }

        inline void set_float(float v) { *data = convert_from_float(v); }
        inline void set_value(element_type v) { *data = v; }
        inline float operator=(float v)
        {
            set_float(v);
            return v;
        }
        // when copying from an Acsr of the same type we don't need to
        // convert to float and back.
        // @@we could also define operator= for other cases, e.g.
        //  int32 from int16, to do the operation without going to float.
        inline AcsrRO operator=(Acsr const &rhs)
        {
            T v = rhs.value();
            set_value(v);
            return AcsrRO(*this);
        }
        inline AcsrRO operator=(AcsrRO const &rhs)
        {
            T v = rhs.value();
            set_value(v);
            return AcsrRO(*this);
        }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};

template <typename T> void PlainInterface<T>::write_floats(void *ptr, const float *srcp, size_t n) const noexcept
{
    T *const dp = static_cast<T *>(ptr);
    for (int i = 0; i < (int)n; i++) {
        dp[i] = convert_from_float(srcp[i]);
    }
}
template <typename T> void PlainInterface<T>::read_floats(float *dstp, const void *ptr, size_t n) const noexcept
{
    T const *const sp = static_cast<T const *>(ptr);
    for (int i = 0; i < (int)n; i++) {
        dstp[i] = convert_to_float(sp[i]);
    }
}

//PlainInterface<float>::convert_from_float: no-op
template <> inline constexpr float PlainInterface<float>::convert_from_float(const float &in)
{
    return in;
}

//PlainInterface<Float16>::convert_from_float: no-op for values in Float16 range, clamp to max otherwise.
template <> inline constexpr Float16 PlainInterface<Float16>::convert_from_float(const float &in)
{
    Float16 const max_as_fp16 = std::numeric_limits<Float16>::max();
    float const max_as_fp32 = static_cast<float>(max_as_fp16);

    if (in > max_as_fp32) return max_as_fp16;
    if (in < -max_as_fp32) return -max_as_fp16;
    return static_cast<Float16>(in);
}

// needs to be defined *after* convert_from_float is specialized
template <typename T> inline void PlainInterface<T>::write_float(void *ptr, const float in) const noexcept
{
    auto p = static_cast<T *>(ptr);
    *p = convert_from_float(in);
}

// these are fully specialized (in tensor.cc)
template <> API_EXPORT void PlainInterface<float>::write_floats(void *ptr, const float *srcp, size_t n) const noexcept;
template <> API_EXPORT void PlainInterface<float>::read_floats(float *dstp, const void *ptr, size_t n) const noexcept;

// make_interface for PlainInterface<T>; easy, just have one
// and return a pointer to it.

template <typename T> struct hnnx::make_interface<PlainInterface<T>> {
    API_EXPORT static PlainInterface<T> const &get_plain_ifc();
    static Interface const *from_odef(Graph &, OutputDef const &odef) { return &get_plain_ifc(); }
    static Interface const *from_deser(Deserializer &dctx) { return &get_plain_ifc(); }
};

extern template class PlainInterface<float>; // in tensor.cc
extern template class PlainInterface<NN_INT32_T>;
/**
 * @class ScaleOffsetInterface
 *
 * @brief A tensor could also have a scale+offset interface
 * This is good for quantization schemes where you want to quantize an arbitrary, possibly asymmetric range.
 * We compute and cache the reciprocal of the scale for conversion from float
 * A default constructor sets the offset to 0 and the scale to 1.0, which would be suitable for integers.
 * The reciprocal of scale should be computed so we don't have to divide.
 */

template <typename T> class ScaleOffsetInterface final : public Interface {
    Interface::qparms qp; // offset, scale, scale_recip;

  public:
    using element_type = T;
    template <typename TX> using interface_other_type = ScaleOffsetInterface<TX>;
    template <typename TX> static inline constexpr T saturate(TX in) { return saturate_cast<T>(in); }
    inline constexpr T convert_from_float(float in) const { return saturate_round<T>(qp.offset + in * qp.scale_recip); }
    inline constexpr float convert_to_float(T in) const
    {
        if constexpr (sizeof(T) <= 2)
            return (float(int(in) - qp.offset)) * qp.scale;
        else
            return (float(in) - qp.offset) * qp.scale;
    }
    virtual inline void write_float(void *ptr, const float in) const noexcept override final
    {
        assert(ptr != nullptr);
        auto p = static_cast<T *>(ptr);
        *p = convert_from_float(in);
    }
    virtual inline float read_float(const void *ptr) const noexcept override final
    {
        assert(ptr != nullptr);
        auto p = static_cast<const T *>(ptr);
        return convert_to_float(*p);
    }
    virtual size_t element_size() const noexcept override final { return sizeof(T); };
    virtual void write_floats(void *ptr, const float *srcp, size_t n) const noexcept override final;
    virtual void read_floats(float *dstp, const void *ptr, size_t n) const noexcept override final;
    // hide the slower implementations of these in the base class...
    inline float get_scale() const noexcept { return qp.scale; }
    inline float get_scale_recip() const noexcept { return qp.scale_recip; }
    inline int32_t get_offset() const noexcept { return qp.offset; }

    virtual bool is_quantized() const noexcept override final { return true; }

    ScaleOffsetInterface(const int offset_, const float scale_) : qp({offset_, scale_, 1.0f / scale_}) {}
    ScaleOffsetInterface(const OutputDef &def) : ScaleOffsetInterface(def.zero_offset, def.stepsize)
    {
        if (def.stepsize == 0.0f) debuglog("Oops: zero stepsize");
    }
    ScaleOffsetInterface() : ScaleOffsetInterface(0, 1.0f) {}
    ScaleOffsetInterface(hnnx::Deserializer &dctx)
    {
        qp.offset = dctx.deserialize_uint32();
        qp.scale = dctx.deserialize_float();
        qp.scale_recip = 1.0f / qp.scale;
    }
    int compare(const ScaleOffsetInterface &rhs) const
    {
        if (qp.offset != rhs.qp.offset) return (qp.offset - rhs.qp.offset);
        if (qp.scale != rhs.qp.scale) return (qp.scale < rhs.qp.scale) ? -1 : 1;
        return 0;
    }
    inline int compare_eq(const ScaleOffsetInterface &rhs) const noexcept
    {
        return qp.offset == rhs.qp.offset && qp.scale == rhs.qp.scale;
    }
    uint32_t interface_hash() const noexcept
    {
        return unsigned(qp.offset) * 0x10661 ^ (image_convert<unsigned, float>(qp.scale) << 1);
    }

  private:
    virtual Interface::qparms const *get_qparms() const noexcept final { return &qp; }

    virtual bool compare_eq_same_type(Interface const *rhs) const noexcept override final
    {
        return compare_eq(*static_cast<ScaleOffsetInterface const *>(rhs));
    }
    // Accessor for ScaleOffsetInterface
    class Acsr;
    class AcsrRO {
      protected:
        T val;
        const ScaleOffsetInterface<T> &interface;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        AcsrRO(void const *data_in, const ScaleOffsetInterface &interface_in)
            : val(*static_cast<T const *>(data_in)), interface(interface_in)
        {
        }
        AcsrRO(AcsrRO const &) = default;
        inline element_type value() const { return val; }
        inline float as_float() const { return interface.convert_to_float(val); }
        inline operator float() const { return as_float(); }
        AcsrRO(Acsr const &a) : val(a.value()), interface(a.interface) {}
    };
    class Acsr {
        friend class AcsrRO;

      protected:
        T *data;
        const ScaleOffsetInterface<T> &interface;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        Acsr(void *data_in, const ScaleOffsetInterface &interface_in)
            : data(static_cast<T *>(data_in)), interface(interface_in)
        {
        }
        Acsr(Acsr const &) = default;
        inline element_type value() const { return *data; }
        inline float as_float() const { return interface.convert_to_float(*data); }
        inline operator float() const { return as_float(); }
        inline void set_float(float v) { *data = interface.convert_from_float(v); }
        inline void set_value(element_type v) { *data = v; }
        inline float operator=(float v)
        {
            set_float(v);
            return v;
        }
        inline float operator=(Acsr const &rhs)
        {
            float const v = rhs.as_float();
            set_float(v);
            return v;
        }
        inline float operator=(AcsrRO const &rhs)
        {
            float const v = rhs.as_float();
            set_float(v);
            return v;
        }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};
template <typename T> void ScaleOffsetInterface<T>::write_floats(void *ptr, const float *srcp, size_t n) const noexcept
{
    ScaleOffsetInterface<T> const intfc = *this;
    T *const dp = static_cast<T *>(ptr);
    for (int i = 0; i < (int)n; i++) {
        dp[i] = intfc.convert_from_float(srcp[i]);
    }
}
template <typename T> void ScaleOffsetInterface<T>::read_floats(float *dstp, const void *ptr, size_t n) const noexcept
{
    ScaleOffsetInterface<T> const intfc = *this;
    T const *const sp = static_cast<T const *>(ptr);
    for (int i = 0; i < (int)n; i++) {
        dstp[i] = intfc.convert_to_float(sp[i]);
    }
}

// make_interface for ScaleOffsetInterface.
template <typename T> struct hnnx::make_interface<ScaleOffsetInterface<T>> {
    // can only declare these here, since we can't see into Graph at this point.
    // Code is in tensor.cc
    API_EXPORT static ScaleOffsetInterface<T> const *from_exemplar(Graph &, ScaleOffsetInterface<T> const &exemplar);
    static Interface const *from_odef(Graph &g, OutputDef const &odef)
    {
        // make an exemplar...
        ScaleOffsetInterface<T> const exemplar(odef);
        return from_exemplar(g, exemplar);
    }
    static Interface const *from_deser(Deserializer &dctx)
    {
        // deserialize the id; p is a pointer to where it is in index,
        // if that's null, it's a new entry and we need to set it.
        Interface const **const p = dctx.deserialize_shared_obj<Interface>();
        if (*p) return *p;
        // make an exemplar, by deserializing
        ScaleOffsetInterface<T> const exemplar(dctx);
        // copy it to crate
        Interface const *const new_p = to_crate(dctx.graph(), exemplar);
        *p = new_p; // for next time it's used
        return new_p;
    }

  protected:
    // put in crate without checking for dups.
    API_EXPORT static ScaleOffsetInterface<T> const *to_crate(Graph &, ScaleOffsetInterface<T> const &exemplar);
};

extern template class ScaleOffsetInterface<uint8_t>; // in tensor.cc
extern template class ScaleOffsetInterface<uint16_t>;

// compare interface. If they are different subclasses, or if either
// or both are 'Interface', this one will be called.
//
// Note: there is no operator !=  - if it's added, it has to be added
// adjacent to all of these or it will be slow. So, just use !(a==b).
//
inline bool operator==(Interface const &a, Interface const &b)
{
    // returns true if a,b are different types; otherwise
    // returns a->compare_eq_same_type(&b).
    return a.compare_equal(b);
}

// Specialize for each subclass.
// All instances of NullInterface are the same.
inline bool operator==(NullInterface const &, NullInterface const &)
{
    return true;
}

// All instances of PlainInterface<T> are the same
template <typename T> inline bool operator==(PlainInterface<T> const &, PlainInterface<T> const &)
{
    return true;
}
// we do need to compare ScaleOffsetInterface
template <typename T> inline bool operator==(ScaleOffsetInterface<T> const &a, ScaleOffsetInterface<T> const b)
{
    if (&a == &b) return true; // same object!
    return a.compare_eq(b);
}

/*
 * We could define a min/max interface, it would be equivalent to ScaleOffset interface.
 * We could enhance the interfaces with overflow flags
 * We could add a scale interface for signed, symmetric values
 */

//////////////////////////////////////////////////////////////////////////////////////////
/// @brief compile-time traits of tensor classes
/// E.g. the construct tensor_traits<TYPE>::element_type will obtain the element_type
/// of any Tensor subclass which has one.
/// Note that tensor_traits<Tensor> has no defined attributes.
///
/// Full list of traits is below
///
/// These are defined in all non-abstract Tensor subclasses:
///  - constexpr DType dtype            // always present (except Tensor,RankedTensor,LayoutTensor); sometimes UNKNOWN
///  - constexpr unsigned rank          // always present (except Tensor); sometimes 0
///  - typedef element_type;            // always present (except Tensor,RankedTensor,LayoutTensor); void in TensorShape
///  - typedef storage_type             // always present (except Tensor,RankedTensor); void in TensorShape
///
/// In LayoutTensor and ConcreteTensor:
///  - typedef layouttensor_type        // The LayoutTensor<> class
///  - typedef layout_type
///  - typedef pad_type                 // Padding<rank> or NoPadding<rank>
///  - constexpr bool has_padding
///  - constexpr bool is_chunked;
///  - constexpr bool is_indirect;      // usually same as is_chunked; always <= is_chunked
///  - typedef raw_type                 // See below [1]
///
/// Only in ConcreteTensor:
///  - constexpr MemoryClass memclass;
///
/// Only in ConcreteTensor, TensorScalar:
///  - typedef interface_type
///
///  [1] raw_type is defined in the classes which have get_raw(...),
///     and is the type which get_raw returns a ref to.
///     For LayoutTensor, it is the same as storage_type; for ConcreteTensor,
///     it is the same as element_type.
///
template <typename TENST> using tensor_traits = typename TENST::traits;

//////////////////////////////////////////////////////////////////////////////////////////

// this is returned by Tensor::get_dtype_intfc()
//
struct DTypeScaleOff {
    DType dtype;
    float scale;
    int offset;
    DTypeScaleOff(DType dt, float sc, int zo) : dtype(dt), scale(sc), offset(zo) {}
    explicit DTypeScaleOff(DType dt) : dtype(dt), scale(1.0f), offset(0) {}
    DTypeScaleOff() : DTypeScaleOff(DType::UNKNOWN) {}
    // construct from an Interface
    API_EXPORT DTypeScaleOff(DType dt, Interface const &) noexcept;
    DTypeScaleOff(DTypeScaleOff const &) = default;
    DTypeScaleOff &operator=(DTypeScaleOff const &) = default;
};

/*
 * Now that we have Interfaces and Accessors, which we will use to give a consistent interface to Tensors,
 * let's work on the actual Tensors
 */

/*
 * @class Tensor
 *
 * @brief This is the abstract base class for Tensors.
 * All tensors allow you to index into them with foo(a,b,c);
 * You can query rank, dim, etc
 *
 * But, you're probably better off with one of the more specific Tensor types for performance,
 * since a lot of the virtual functions become trivial for the compiler if they can be inlined.
 *
 */

class Tensor {
  public:
    enum class clone_mode {
        duplicate,
        UNUSED_persistent,
    };

    // Use with 'dims' to query dimension sizes by name e.g. auto [h, d] = tensor.dims(Tensor::HEIGHT, Tensor::DEPTH)
    enum dimensions { BATCH, HEIGHT, WIDTH, DEPTH, CHANNEL };

    // These functions are not really part of the interface, but we use them to implement operator()
    // EJP: FIXME: alternative strategies for implementing these, flat index may not make sense
    // for all types of layout.  We could pass rank,coords to generic_accessor
    virtual Interface const &interface() const noexcept = 0;
    GenericAccessor generic_accessor(void *p) noexcept { return GenericAccessor(p, interface()); }
    GenericAccessorRO generic_accessor_ro(void const *p) const noexcept { return GenericAccessorRO(p, interface()); }
    struct traits { // empty
    };
    virtual const char *true_name() const { return typeid(*this).name(); }
    Tensor(const Op *producer_in) {}
    Tensor(const Op *producer_in, hnnx::Deserializer &) {}
    Tensor(const Tensor &old, hnnx::Allocator *allocator, clone_mode) {}
    virtual ~Tensor(){}; // virtual destructor
    virtual const size_t rank() const noexcept = 0; // What's the rank of this tensor?
    virtual const size_t dim(size_t index) const noexcept = 0; // What's the length of some dimension?
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept = 0; // return rank, and address of dims[0..n-1]
    // this is the first virtual method defined externally.
    API_EXPORT virtual uint32_t find_content_hash() const noexcept; // find 'content hash' of the data.

  protected:
    // Note, this is a const method returning a non-const pointer;
    // but we only allow it to publicly return a non-const
    // pointer when used in non-const wrapper methods.
    virtual void *element_addr(size_t rank, SIdx const coords_in[]) const noexcept = 0;

  public:
    // element_ptr on insufficiently specialized class gives the result as a void *.
    inline ALWAYSINLINE void const *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (void const *)element_addr(rank, coords);
    }
    inline ALWAYSINLINE void *element_ptr(size_t rank, const SIdx coords[]) { return element_addr(rank, coords); }

    std::tuple<size_t, size_t, size_t, size_t> get_dims_4() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims(); // virtual call
        if (n != 4) throw std::runtime_error("rank not 4");
        return std::make_tuple(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
    // this is a common case.
    std::tuple<size_t, size_t> get_dims_1_2() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims(); // virtual call
        if (n < 3) throw std::runtime_error("rank not >=3");
        return std::make_tuple(ptr[1], ptr[2]);
    }

    constexpr std::array<size_t, 4> dims() const
    { // make compatible with typical concrete tensor.
        std::array<size_t, 4> ret{0};
        for (int i = 0; i < 4; i++) {
            ret[i] = dim(i);
        }
        return ret;
    }

    template <typename... T> const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        std::array<size_t, sizeof...(indices)> dim_sizes = {dim(indices)...};
        return dim_sizes;
    }

    virtual DTypeScaleOff get_dtype_intfc() const noexcept = 0;
    // if you need more than one of these, it is recommended to unpack
    // the result from get_dtype_intfc()
    DType get_dtype() const { return get_dtype_intfc().dtype; }
    float get_interface_scale() const { return get_dtype_intfc().scale; }
    NN_INT32_T get_interface_offset() const { return get_dtype_intfc().offset; }
    API_EXPORT OutputDef gen_output_def() const;

    template <typename... ind_types> inline GenericAccessorRO operator()(ind_types... inds) const
    {
        const std::array<SIdx, sizeof...(ind_types)> indarr = {static_cast<SIdx>(inds)...};
        return this->generic_accessor_ro(element_addr(sizeof...(ind_types), indarr.data()));
    }
    template <typename... ind_types> inline GenericAccessor operator()(ind_types... inds)
    {
        const std::array<SIdx, sizeof...(ind_types)> indarr = {static_cast<SIdx>(inds)...};
        return this->generic_accessor(element_addr(sizeof...(ind_types), indarr.data()));
    }

    /*
     * Returned by virtual method get_tensor_format_code:
     *
     *                                   (General)           (Shape)             (Scalar)
     *  Bits  3:0  dtype code             x                  0 = UNKNOWN          x
     *  Bits  5:4  (reserved, zero)
     *  Bits  7:6  log2(element_size)     x                  0                    x
     *  Bits 11:8  rank                   x                  x                    0
     *  Bits 15:12 (reserved, 0)
     *  Bit  16     is_tcm                x                  0                    0
     *  Bit  17     is_quantized          x                  0                    x
     *  Bit  18     is_indirect           x                  0                    0
     *  Bit  19     is_chunked            x                  0                    0
     *  Bit  20     is_not_flat           x                  0                    0
     *  Bits  27:31	 (reserved, 0)
     *  Bits 31:28  mode                  tensMODE_general   tensMODE_shape       tensMODE_scalar
	 *-------------------------------------
     * Returned by get_tensor_info():
     * This is a bit weird, due to a legacy bug, but I'm restating it as below,
     * which remains compatible:
     *
     *  For Concrete tensor:
     *      Bits 3:0    DType
     *      Bits 7:4    '0001'
     *      Bits 11:8   rank
     *      Bits 15:12  '0000'
     *      Bits 19:16  memory class
     *      Bits 27:20  zero
     *      Bits 31:28  tensMODE_general
     *
     *  For Shape and Scalar tensors: Bits 31:28 contain tensMODE_shape, or tensMODE_scalar; others bits ar 0.
     *  Classes which cannot be serialized return 0 in the upper 4 bits.
     */
    enum {
        tformat_dtype_shift = 0u,
        tformat_dtype_mask = 0xFu,
        tformat_log2sz_shift = 6u,
        tformat_log2sz_mask = 3u,
        tformat_rank_shift = 8u,
        tformat_rank_mask = 0xFu,
        tformat_is_tcm = 1u << 16u,
        tformat_is_quantized = 1u << 17u,
        tformat_is_indirect = 1u << 18u,
        tformat_is_chunked = 1u << 19u,
        tformat_is_not_flat = 1u << 20u,
        tformat_tmode_shift = 28u,
        tformat_tmode_mask = 0xFu,
    };

  protected:
    template <typename IFC> static inline constexpr uint32_t formatcode_for_interface()
    {
        constexpr DType dt = dtype_of_type<IFC>();
        uint32_t result = unsigned(dt);
        constexpr unsigned elbytes = sizeof(typename IFC::element_type);
        constexpr unsigned log2sz = (elbytes == 8) ? 3 : (elbytes == 4) ? 2 : (elbytes == 2) ? 1 : 0;
        static_assert(elbytes == (1u << log2sz));
        result |= log2sz << tformat_log2sz_shift;
        if (dtype_traits<dt>::is_quant) result |= tformat_is_quantized;
        return result;
    }
    template <typename TRAITS> static inline constexpr uint32_t formatcode_for_general()
    {
        constexpr unsigned rankval = TRAITS::rank;
        uint32_t result = formatcode_for_interface<typename TRAITS::interface_type>();
        result |= (rankval << tformat_rank_shift);
        if (TRAITS::memclass == MemoryClass::TCM) result |= tformat_is_tcm;
        if (TRAITS::is_indirect) result |= tformat_is_indirect;
        if (TRAITS::is_chunked) result |= tformat_is_chunked;
        if (!std::is_base_of_v<FlatMemoryLayout<rankval>, typename TRAITS::layout_type>) {
            result |= tformat_is_not_flat;
        }
        return (hnnx::SerOpsInterface::tensMODE_general << tformat_tmode_shift) | result;
    }

    template <unsigned RANK> static inline constexpr uint32_t formatcode_for_shape()
    {
        static_assert(RANK <= tformat_rank_mask);
        return (hnnx::SerOpsInterface::tensMODE_shape << tformat_tmode_shift) | (RANK << tformat_rank_shift);
    }
    template <typename IFC> static inline constexpr uint32_t formatcode_for_scalar()
    {
        return (hnnx::SerOpsInterface::tensMODE_scalar << tformat_tmode_shift) | formatcode_for_interface<IFC>();
    }

    static inline constexpr uint32_t pack_tensor_info(DType type, uint32_t rank, MemoryClass mclass)
    {
        uint32_t tinfo = 0x10;
        tinfo |= static_cast<uint32_t>(type) & 0xFu;
        tinfo |= (rank & 0xFu) << 8u;
        tinfo |= (static_cast<uint32_t>(mclass) & 0xF) << 16u;
        tinfo |= hnnx::SerOpsInterface::tensMODE_general << tformat_tmode_shift;
        return tinfo;
    }

  public:
    API_EXPORT virtual uint32_t get_tensor_info() const noexcept; // returns 0;
    API_EXPORT virtual uint32_t get_tensor_format_code() const noexcept; // returns 0;
    // returns false if the dims are the same; true if different, or maybe different.
    virtual bool set_dims(const size_t dims[]) = 0; // Set the shape of the tensor
    virtual bool set_dims(const Tensor &prototype) = 0; // Set the shape of the tensor same as another.

    inline void allocate(hnnx::Allocator &allocator, unsigned options = 0) { return allocate_func(allocator, options); }

    /* EJP: FIXME: temporary functions */
    /*
	 * Some of these functions are convenient for now, but don't necessarily
	 * need to live for a long time if we find better ways of doing things.
	 */
    virtual void *raw_data() noexcept = 0; // Get pointer to raw data
    void const *raw_data_const() const noexcept { return const_cast<Tensor *>(this)->raw_data(); }
    virtual void set_raw_data_despite_danger(void *buffer)
    {
        assert(!"Invalid to set raw pointer on this type of tensor");
    }
    virtual size_t total_storage_elements() const = 0;
    virtual size_t total_storage_bytes() const = 0;
    const char *truetype() const noexcept { return typeid(*this).name(); }

    // Append the set of allocated memory blocks to blocklist.
    API_EXPORT void get_memory_blocks(hnnx::blockid_set_t &blocklist, int mc_sel = -1) const;
    inline void get_memory_blocks(hnnx::blockid_set_t &blocklist, MemoryClass mc) const
    {
        get_memory_blocks(blocklist, int(mc));
    }
    // return the set of memory blocks
    API_EXPORT hnnx::blockid_set_t get_memory_blocks(int mc_sel = -1) const;
    inline hnnx::blockid_set_t get_memory_blocks(MemoryClass mc) const { return get_memory_blocks(int(mc)); }

    // Supply the allocated memory blocks to the enumerator.
    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const = 0;

    // The 'ef' parameter to these functions is a callable (function, lambda, std::function...)
    // compatible with MemBlockEnumerator::supply_blocks_func
    template <typename ENFUNC> inline void enum_memory_blocks_withfunc(ENFUNC &&ef) const
    {
        hnnx::MemBlockEnumWrapper<std::remove_reference_t<ENFUNC>> enumer(std::forward<ENFUNC>(ef));
        this->enum_memory_blocks(enumer);
    }
    // The 'rf' parameter to these functions is a callable (function, lambda, std::function...)
    // .. called as ( Tensor const *, void *old_blkid) -> void *new_blkid
    template <typename REPLFUNC> inline void replace_memory_blocks_withfunc(REPLFUNC &&rf)
    {
        hnnx::MemBlockReplBlockWrapper<std::remove_reference_t<REPLFUNC>> enumer(std::forward<REPLFUNC>(rf));
        this->enum_memory_blocks(enumer);
    }
    // this is passed a map<void*,void*> or any similar type with find() and end(),
    // and uses it to edit the blocks in the tensor.
    template <typename MAPTYPE> inline void replace_memory_blocks_withmap(MAPTYPE const &map)
    {
        replace_memory_blocks_withfunc([&map](Tensor const *, void *oldid) {
            auto found_at = map.find(oldid);
            return (found_at != map.end()) ? found_at->second : oldid;
        });
    }

    void serialize(hnnx::SerOpsInterface &sctx) const { sctx.tensor_serialize(this); }
    // The same tensor in the same layout, but with persistent storage.

    API_EXPORT std::unique_ptr<Tensor> persistent_clone(hnnx::Allocator *allocator, bool zoneb = false) const;
    // same thing, but does refcounts in 'zone B'
    inline std::unique_ptr<Tensor> persistent_clone_Op(hnnx::Allocator *allocator) const
    {
        return persistent_clone(allocator, true);
    }
    // similar in effect to persistent_clone_Op, but can onlt be applied to
    // existing persistent tensors; and only copies the tensor, not the data.

    API_EXPORT std::unique_ptr<Tensor> shallow_clone_Op(hnnx::Allocator *allocator) const;

    // decref the ref counts of any contained blocks (all must be persistent)
    API_EXPORT void persistent_decref(hnnx::Allocator *allocator, bool zoneb = false) const;
    // same thing, but does refcounts in 'zone B'
    inline void persistent_decref_Op(hnnx::Allocator *allocator) const { return persistent_decref(allocator, true); }

    // a 'duplicate' - same type,layout,dims; references the same
    // memory block(s) (where applicable).
    std::unique_ptr<Tensor> duplicate_clone(hnnx::Allocator *allocator) const
    {
        return reallocate_clone(allocator, true);
    }
    // do a 'reallocate clone': the new tensor is the same type, layout, dims
    // but the block table is zeroed.
    // If dup=true, this is the same as duplicate_clone.
    API_EXPORT std::unique_ptr<Tensor> reallocate_clone(hnnx::Allocator *allocator, bool dup = false) const;

    // 'compare' in the base class:
    //   - if the types are different, return -1 or 1 depending on that.
    //   - otherwise call protected virtual compare_sametype(), which can then use static_cast
    //     to downcast (and doesn't need to recurse back to the base).

    int compare(const Tensor *rhs) const
    {
        Tensor const *const lhs = this;
        std::type_info const &lhs_type = typeid(*lhs);
        std::type_info const &rhs_type = typeid(*rhs);
        if (lhs_type == rhs_type) {
            return lhs->compare_sametype(rhs);
        } else {
            return lhs_type.before(rhs_type) ? -1 : 1;
        }
    }

    virtual uint64_t get_checksum() const { return 0LL; };
    // these only work on specific types; in others, you inherit the base class implementation
    // which raises a runtime error. You can use tile_support() to find out if support exists
    API_EXPORT virtual void const *read_tile(unsigned flags, void *buffer, size_t b, int h, int w, int d) const;
    API_EXPORT virtual void write_tile(unsigned flags, void const *buffer, size_t b, int h, int w, int d);
    enum {
        tile_8bit = 1, // set when the data is 8 bit and the tensor supports tile access
        tile_16bit = 2, // set when the data is 16 bit and the tensor supports tile access
        tile_32bit = 4, // set when the data is 32 bit and the tensor supports tile access
        tile_any = (1 + 2 + 4), // one of these bits is set if there is any support
        tile_fast = 16, // set only when one of the XXbit is set, and the support is vector accelerated.
        tile_direct = 32 // set only when when 'fast' is set, and a direct mapping is possible
    };

    API_EXPORT virtual unsigned tile_support_bits() const;
    inline bool tile_support() const { return (tile_support_bits() & tile_any) != 0; }
    inline bool tile_support_fast() const { return (tile_support_bits() & tile_fast) != 0; }
    inline bool tile_support_direct() const { return (tile_support_bits() & tile_direct) != 0; }
    // this is currently a wrapper on tile_write, which inserts the 'write_strategy' flag, and suppresses broadcast
    // and copy flags. It may change to a separate virtual func.
    // (this is defined as an inline, in tile_extract.h).
    void *write_tile_strategy(unsigned flags, void *buffer, size_t b, int h, int w, int d);

    API_EXPORT static uint32_t content_hash_data(void const *, size_t nbytes, bool is_float) noexcept;
    API_EXPORT static uint32_t content_hash_data_indirect(uint32_t inhash, void **blocks, unsigned nblocks,
                                                          size_t blockbytes, bool is_float) noexcept;

    API_EXPORT static uint32_t build_hash(size_t const *dims, int n, uint32_t previous) noexcept;

    struct tensor_blockinfo {
        void **blkptrs; // pointer to block table (nullptr if no blocks)
        // shapepp is a pointer to the shape pointer (where applicable; otherwise null). If a clone
        // is done, it points to the field in the cloned tensor.
        hnnx::ShapeFlags const *const *shapepp;
        size_t nblocks; // number of blocks
        size_t blocksize; // size of block, in bytes
        DType dtype;
        MemoryClass mclass;
        bool is_indirect; // indicates that the layout is indirect.
        bool is_chunked; // indicates that the layout is chunked;
        void setup(DType dt = DType::UNKNOWN, MemoryClass mc = MemoryClass::Default)
        {
            blkptrs = nullptr;
            shapepp = nullptr;
            nblocks = 0;
            blocksize = 0;
            dtype = dt;
            mclass = mc;
            is_indirect = false;
            is_chunked = false;
        }
    };
    inline void get_tensor_blockinfo(tensor_blockinfo *infop) const { clone_util(nullptr, nullptr, infop); }

    // deserialize a single block pointer for a contiguous tensor.
    static void *deserialize_block_pointer(hnnx::Deserializer &dctx);
    // deserialize an indirect blocktable, return pointer.
    static void **deserialize_blocktable(hnnx::Deserializer &dctx, unsigned const nblocks);

  protected:
    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) = 0;
    virtual int compare_sametype(const Tensor *rhs) const = 0;

    // clone_util is an overburdened virtual function, which performs duplicate_clone almost directly,
    // and provides other info by which the other clone methods, and the decref methods,
    // can all be done generically in the base class.
    //
    // - If tensp != null, it will create a duplicate_clone, and store it at *tensp;
    // - If infop != null, it will fill in *infop with the tensor info.
    // If *both* are not null, then infop->blkptrs will point to the block table in the
    //  original tensor, and the return value is the block pointer in the new tensor.
    //  Otherwise the return value is null (and it will be null in any case, if the tensor
    //  has no blocks).
    //
    //
    // Note: allocator may be null if tensp is null.

    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              tensor_blockinfo *infop) const = 0;
};

// A FakeTensor is intended as an intermediate base for special subclasses which
// need to be based on Tensor but don't need to support most of the interface.
// subclassing should be done in .cc files or private headers where possible.
//
// All of the abstract 'virtual=0' methods (other than get_dtype) are overridden here;
// many (those shown as protected) will all throw exceptions if called; the others do
// null things as shown.
// So when you subclass, just override whatever ones you need and leave the rest.
//
// In particular, get_dtype() returns DType::None.
//
class FakeTensor : public Tensor {
  public:
    FakeTensor(const Op *producer_in) : Tensor(producer_in) {}
    API_EXPORT FakeTensor(const Op *producer_in, hnnx::Deserializer &);

  protected:
    // all will throw exception if called
    API_EXPORT virtual void *element_addr(size_t rank, SIdx const coords_in[]) const noexcept override;
    API_EXPORT virtual Interface const &interface() const noexcept override;
    API_EXPORT virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override;
    API_EXPORT virtual void *raw_data() noexcept override;
    API_EXPORT virtual size_t total_storage_elements() const override;
    API_EXPORT virtual size_t total_storage_bytes() const override;
    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         Tensor::tensor_blockinfo *infop) const override;
    API_EXPORT virtual int compare_sametype(const Tensor *rhs) const override;

  public:
    // defined as shown
    API_EXPORT virtual const size_t rank() const noexcept override; //->0
    API_EXPORT virtual const size_t dim(size_t index) const noexcept override; //->0
    API_EXPORT virtual std::pair<size_t const *, size_t> get_dims() const noexcept override; //->{null,0}
    API_EXPORT virtual bool set_dims(const size_t dims[]) override; // -> false
    API_EXPORT virtual bool set_dims(const Tensor &prototype) override; // ->false
    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override; // nothing

    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept override; // { return DTypeScaleOff(None); }
};

/**
 * @class RankedTensor
 *
 * @brief Almost as abstract as Tensor, but we template the Rank.
 * This allows us to have compile-time checking of the operator(), as well as a public Rank that is static constexpr
 * so it doesn't take a lot of space or performance...
 * The other benefit here is that we can specify RankedTensor<4> to have a fairly generic tensor,
 * but enforce the number of dimensions of the tensor.
 */

template <unsigned TRank> class RankedTensor : public Tensor {
  public:
    struct traits {
        static constexpr unsigned Rank = TRank;
    };

    RankedTensor(const Op *producer_in) : Tensor(producer_in) {}
    RankedTensor(const Op *producer_in, hnnx::Deserializer &dctx) : Tensor(producer_in, dctx) {}
    RankedTensor(const RankedTensor &old, hnnx::Allocator *allocator, clone_mode cmode) : Tensor(old, allocator, cmode)
    {
    }
    static constexpr auto Rank = TRank;
    virtual inline const size_t rank() const noexcept override final { return Rank; }
    template <typename... ind_types> inline GenericAccessorRO operator()(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const SIdx indarr[] = {static_cast<SIdx>(inds)...};
        return this->generic_accessor_ro(this->element_addr(Rank, indarr));
    }
    template <typename... ind_types> inline GenericAccessor operator()(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const SIdx indarr[] = {static_cast<SIdx>(inds)...};
        return this->generic_accessor(this->element_addr(Rank, indarr));
    }
};

/**
 * @class TensorShape
 *
 * @brief This is a tensor that just has a shape, no memory or type or anything.
 * This needs to be non-abstract
 *
 * EJP: FIXME: should we really use this, or just use Const? Or special like-Const op?
 * EJP: FIXME: Performance is not so criticial here, we need it to respect the interface
 * but we really want to make this convenient representation and be formable from an OutputDef
 *
 * TensorShape should already be canonized by the nature of being a const op.
 * We might be able to share TensorShapes shapes and Tensor shapes, but it seems unnecessary.
 */

template <unsigned TRank> class TensorShape : public RankedTensor<TRank> {
    using Parent = RankedTensor<TRank>;

  protected:
    API_EXPORT static constexpr NullInterface null_interface{};
    // These functions are not really part of the interface, but we need them to implement operator()
    virtual void *element_addr(size_t rank, SIdx const coords_in[]) const noexcept override final { return nullptr; }
    virtual Interface const &interface() const noexcept override { return null_interface; }

  public:
    const char *true_name() const override { return type_name<TensorShape<TRank>>(); };

    using Parent::Rank;
    struct traits {
        using element_type = void;
        using storage_type = void;
        static constexpr DType dtype = DType::UNKNOWN;
        static constexpr unsigned rank = TRank;
    };

    //using Shape_t = Shape<Rank>;
    //const Shape_t *shape;
    const std::array<size_t, Rank> shape;
    virtual const size_t dim(size_t index) const noexcept override final { return shape[index]; }
    const std::array<size_t, Rank> &dims() const { return shape; };
    template <typename... T> const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        std::array<size_t, sizeof...(indices)> dim_sizes = {dim(indices)...};
        return dim_sizes;
    }
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept override
    {
        return std::pair<size_t const *, size_t>(&shape[0], Rank);
    }
    virtual DTypeScaleOff get_dtype_intfc() const noexcept override { return DTypeScaleOff(); }

    virtual uint32_t get_tensor_format_code() const noexcept override { return Tensor::formatcode_for_shape<Rank>(); }
    virtual uint32_t get_tensor_info() const noexcept override
    {
        return hnnx::SerOpsInterface::tensMODE_shape << Tensor::tformat_tmode_shift;
    }

    // Optional, but maybe helpful?
    virtual bool set_dims(const size_t dims[]) override
    {
        static_assert("Shapes are immutable");
        return true;
    } // immutable
    virtual bool set_dims(const Tensor &prototype) override
    {
        static_assert("Shapes are immutable");
        return true;
    } // immutable
    // EJP: FIXME: temporary functions
    virtual void *raw_data() noexcept override
    {
        return nullptr;
    } // Allocate storage ourselves instead of fancy memory allocator
    virtual size_t total_storage_elements() const override { return 0; }
    virtual size_t total_storage_bytes() const override { return 0; }
    TensorShape(const Op *producer_in, const OutputDef &def, Graph &graph_in)
        : Parent(producer_in), shape(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0]))
    {
    }
    TensorShape(const Op *producer_in, hnnx::Deserializer &dctx)
        : Parent(producer_in, dctx), shape(dctx.deserialize_uint32_array_sizet<Rank>())
    {
    }
    TensorShape(const TensorShape &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode)
        : Parent(old, allocator, cmode), shape(old.shape)
    {
    }

    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override { return; }

    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override final {}

    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              Tensor::tensor_blockinfo *infop) const override
    {
        if (tensp) *tensp = std::make_unique<TensorShape>(*this, allocator, Tensor::clone_mode::duplicate);
        if (infop) infop->setup();
        return nullptr;
    }

  protected:
    virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        auto *rhs = static_cast<const TensorShape *>(rhs_in);
        for (int i = 0; i < Rank; i++) {
            int const dimdiff = this->shape[i] - rhs->shape[i];
            if (dimdiff != 0) return dimdiff;
        }
        return 0;
    }
    virtual uint32_t find_content_hash() const noexcept override
    {
        return Tensor::build_hash(&shape[0], Rank, 0x113014);
    }
    static const char *code_to_type_name() { return TensorTypeStruct<TensorShape<TRank>>::name; }
};

/*
 * I think we should have a Scalar Constant
 * * Immutable shape
 * * Rank 0
 * * Coords ignored
 * * Dim(x) == 0
 * * Templated type / interface?
 */

//
// Tensor Scalar depending on DType

template <DType DT> class TensorSclrDT : public Tensor {
  protected:
    using T = typename dtype_traits<DT>::element_type;
    using Interface_t = std::conditional_t<dtype_traits<DT>::is_quant, ScaleOffsetInterface<T>, PlainInterface<T>>;
    using Accessor_t = typename Interface_t::Accessor;
    using Const_Accessor_t = typename Interface_t::AccessorRO;
    Interface_t interface_inst;

  public:
    virtual Interface_t const &interface() const noexcept override final { return interface_inst; }
    inline float interface_scale() const { return interface_inst.get_scale(); }
    inline float interface_scale_recip() const { return interface_inst.get_scale_recip(); }
    inline int32_t interface_offset() const { return interface_inst.get_offset(); }

    virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return Tensor::formatcode_for_scalar<Interface_t>();
    }
    virtual uint32_t get_tensor_info() const noexcept override
    {
        return hnnx::SerOpsInterface::tensMODE_scalar << tformat_tmode_shift;
    }

  protected:
    // EJP: FIXME: this should just be the value, but then GenericAccessor constructor
    // complains about const value going to a const Accessor where the constructor in
    // const Accessor is written to have a normal void pointer input... sigh.
    T value;
    // These functions are not really part of the interface, but we need them to implement operator()
    virtual void *element_addr(size_t rank, SIdx const coords_in[]) const noexcept override final
    {
        return (void *)&value;
    }

  public:
    const char *true_name() const override { return type_name<TensorSclrDT<DT>>(); };

    struct traits {
        using element_type = T;
        using storage_type = typename dtype_traits<DT>::storage_type;
        using interface_type = Interface_t;
        static constexpr DType dtype = DT;
        static constexpr unsigned rank = 0;
    };

    virtual const size_t rank() const noexcept override { return 0; } // What's the rank of this tensor?
    virtual const size_t dim(size_t index) const noexcept override { return 1; } // What's the length of some dimension?
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept override
    {
        return std::pair<size_t const *, size_t>(nullptr, 0);
    }
    static constexpr DType dtype = dtype_of_type<Interface_t>();
    virtual DTypeScaleOff get_dtype_intfc() const noexcept override { return DTypeScaleOff(dtype, interface()); }

    // Optional, but maybe helpful?
    virtual bool set_dims(const size_t dims[]) override
    {
        static_assert("Scalar dims are immutable");
        return true;
    } // immutable
    virtual bool set_dims(const Tensor &prototype) override
    {
        static_assert("Scalar dims are immutable");
        return true;
    } // immutable
    // EJP: FIXME: temporary functions
    virtual void *raw_data() noexcept override final
    {
        return &value;
    } // Allocate storage ourselves instead of fancy memory allocator
    const void *raw_data() const noexcept
    {
        return &value;
    } // Allocate storage ourselves instead of fancy memory allocator
    virtual size_t total_storage_elements() const override { return 0; }
    virtual size_t total_storage_bytes() const override { return 0; }
    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override final {}
    TensorSclrDT(const Op *producer_in, T value_in) : Tensor(producer_in), value(value_in)
    {
        static_assert(!dtype_traits<DT>::is_quant, "FIXME: need different constructor");
    }
    TensorSclrDT(const Op *producer_in, hnnx::Deserializer &dctx)
        : Tensor(producer_in, dctx), interface_inst(dctx), value(dctx.deserialize_type<T>())
    {
    }
    TensorSclrDT(const TensorSclrDT &old, hnnx::Allocator *allocator, clone_mode cmode)
        : Tensor(old, allocator, cmode), interface_inst(old.interface_inst), value(old.value)
    {
    }

    template <typename... ind_types> inline const Const_Accessor_t operator()(ind_types... inds) const
    {
        return Const_Accessor_t((void *)&value, this->interface());
    }
    template <typename... ind_types> inline Accessor_t operator()(ind_types... inds)
    {
        return Accessor_t(&value, this->interface());
    }
    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override { return; }

    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              Tensor::tensor_blockinfo *infop) const override
    {
        if (tensp) *tensp = std::make_unique<TensorSclrDT>(*this, allocator, clone_mode::duplicate);
        if (infop) infop->setup(DT);
        return nullptr;
    }

  protected:
    virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        auto *rhs = static_cast<const TensorSclrDT *>(rhs_in);
        if (this->value < rhs->value) return -1;
        if (this->value == rhs->value) return 0;
        return 1;
    }
    virtual uint32_t find_content_hash() const noexcept override
    {
        uint32_t const h = interface().interface_hash() ^ mulu32_modular(unsigned(DT), 0x107301);
        return mulu32_modular(h, 0x104301) ^ content_hash_data(&this->value, sizeof(T), dtype_traits<DT>::is_float);
    }
    static const char *code_to_type_name() { return TensorTypeStruct<TensorSclrDT<DT>>::name; }
};
// Tensor Scalar depending on type (assuming PlainInterface)

template <typename T> using TensorScalar = TensorSclrDT<dtype_of_type<PlainInterface<T>>()>;

///////////////////////////////////////////
// this is contained within LayoutTensor, and implements the block pointer,
// or pointers.
// The first parameter indicates if the layout is indirect; we specialize
// the whole class on true vs. false.
// The remaining template parms are the same as those in the LayoutTensot containing
// it.

/// >> for contiguous tensors

template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_contig {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};

    storage_type *bulk_data;

    inline layout_mem_contig(Shape_t const *shp, Graph &graph_in) : bulk_data(){};

    // duplicate clone from another
    inline layout_mem_contig(Shape_t const *shp, layout_mem_contig const &other, hnnx::Allocator *alloc,
                             Tensor::clone_mode cmode)
        : bulk_data(other.bulk_data)
    {
    }

    // construct from deserialize
    layout_mem_contig(Shape_t const *, hnnx::Deserializer &dctx)
        : bulk_data((storage_type *)Tensor::deserialize_block_pointer(dctx))
    {
    }

    // this implements raw_data in the containing tensor
    inline ALWAYSINLINE void *raw_data() const noexcept { return (void *)bulk_data; }

    // this implements set_raw_data_despite_danger(void *buffer) override final { bulk_data = static_cast<T *>(buffer); }
    inline ALWAYSINLINE void set_raw_data_despite_danger(void *buffer)
    {
        bulk_data = static_cast<storage_type *>(buffer);
    }

    // this implements element_addr in the containing tensor.
    ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[]) const noexcept
    {
        //assert(rank == Rank);
        const std::array<size_t, Rank> padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const offset = layout.linear_offset(padded_coords, shp->max_dims);
        return (void *)&bulk_data[offset];
    }

    // get pointer to block table, and length
    inline ALWAYSINLINE void **get_block_list_ptr() const { return (void **)&bulk_data; }
    static inline ALWAYSINLINE size_t get_block_list_len(Shape_t const *shp) { return 1; }
    // block size for allocation
    inline ALWAYSINLINE static size_t get_elements_per_block(Shape_t const *shp)
    {
        return std::accumulate(shp->max_dims.cbegin(), shp->max_dims.cend(), 1, std::multiplies<size_t>());
    }
    // find the address of the block pointer containing the specified coords.
    // (not used for contig. tensor, but this is reasonable impl).
    inline storage_type **block_ptr_addr(Shape_t const *shape, std::array<SIdx, Rank> coords) const
    {
        return &bulk_data;
    }
    // dummy for this
    inline void realloc_blocktab(hnnx::Allocator *alloc, Shape_t const *old_shape, Shape_t const *new_shape)
    {
        bulk_data = nullptr;
    }

    // compare memory (raw compare)
    int compare_memory(Shape_t const *shp, layout_mem_contig const &rhs) const
    {
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        return memcmp(bulk_data, rhs.bulk_data, len);
    }
    // find content hash of memory.
    //
    uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        return mulu32_modular(oldhash, 0x223131) ^ Tensor::content_hash_data(bulk_data, len, is_float);
    }
};

/// >> for indirect tensors
namespace indirect_layout_mem {
inline void **make_blocktab(size_t n_blocks, Graph &graph_in)
{
    return hnnx::graph_crate(graph_in)->alloc_array_zero<void *>(n_blocks);
}

inline void **make_blocktab_for_overwrite(const size_t n_blocks, Graph &graph_in)
{
    return hnnx::graph_crate(graph_in)->alloc_array<void *>(n_blocks);
}

// TODO: make this not inline.
inline int compare_indirect_blocks(void **ptr_a, void **ptr_b, size_t nblocks, size_t blocklen)
{
    for (size_t i = 0; i < nblocks; i++) {
        int const cmp = memcmp(ptr_a[i], ptr_b[i], blocklen);
        if (cmp != 0) return cmp;
    }
    return 0;
}
} // namespace indirect_layout_mem

//  layout_mem for indirect.
template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_indirect {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};

    storage_type **blocktab;

    // construct table
    layout_mem_indirect(Shape_t const *shp, Graph &graph_in)
        : blocktab((storage_type **)indirect_layout_mem::make_blocktab(layout.num_blocks(shp->max_dims), graph_in))
    {
    }
    // duplicate clone from another
    layout_mem_indirect(Shape_t const *shp, layout_mem_indirect const &other, hnnx::Allocator *alloc,
                        Tensor::clone_mode cmode)
        : blocktab()
    {
        unsigned const nblocks = layout.num_blocks(shp->max_dims);
        blocktab = (storage_type **)indirect_layout_mem::make_blocktab_for_overwrite(nblocks, alloc->graph);
        std::memcpy(blocktab, other.blocktab, sizeof(void *) * nblocks);
    }
    // construct from deserialize
    layout_mem_indirect(Shape_t const *shp, hnnx::Deserializer &dctx) : blocktab()
    {
        blocktab = (storage_type **)Tensor::deserialize_blocktable(dctx, layout.num_blocks(shp->max_dims));
    }

    // this implements raw_data in the containing tensor
    inline ALWAYSINLINE void *raw_data() const noexcept { return (void *)blocktab[0]; }
    // this implements set_raw_data_despite_danger(void *buffer) override final { bulk_data = static_cast<T *>(buffer); }
    inline void set_raw_data_despite_danger(void *buffer)
    {
        assert(!"Invalid to set raw pointer on this type of tensor");
    }

    // this implements element_addr in the containing tensor.
    ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[]) const noexcept
    {
        assert(rank == Rank);
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const block_offset = layout.chunk_offset(padded_coords, shp->max_dims);
        size_t const block_idx = layout.chunk_index(padded_coords, shp->max_dims);
        return (void *)&blocktab[block_idx][block_offset];
    }
    // get pointer to block table, and length
    inline ALWAYSINLINE void **get_block_list_ptr() const { return (void **)blocktab; }
    static inline ALWAYSINLINE size_t get_block_list_len(Shape_t const *shp)
    {
        return layout.num_blocks(shp->max_dims);
    }
    // block size for allocation
    inline ALWAYSINLINE static size_t get_elements_per_block(Shape_t const *) { return layout.chunk_total; }
    // find the address of the block pointer containing the specified coords.
    inline storage_type **block_ptr_addr(Shape_t const *shape, std::array<SIdx, Rank> coords) const
    {
        std::array<size_t, Rank> const padded_coords = pad.pad_coords(coords, shape->pad);
        size_t const block_idx = layout.chunk_index(padded_coords, shape->max_dims);
        return &blocktab[block_idx];
    }
    // reallocate for change from old_shape to new_shape (typically just the padding
    // is changed) and zero the blocktab. If the shape is not actually changed, or if
    // the blocktab isn't larger than before, we keep the old one, but we still clear it.
    inline void realloc_blocktab(hnnx::Allocator *alloc, Shape_t const *old_shape, Shape_t const *new_shape)
    {
        unsigned const nblocks = layout.num_blocks(new_shape->max_dims);
        if (old_shape != new_shape) {
            unsigned const old_nblocks = layout.num_blocks(old_shape->max_dims);
            if (nblocks > old_nblocks) { // need reallocate.
                blocktab = (storage_type **)indirect_layout_mem::make_blocktab(nblocks, alloc->graph);
                return; // already zeroed
            }
        }
        ::memset(blocktab, 0, nblocks * sizeof(void *));
    }

    // compare memory (raw compare)
    int compare_memory(Shape_t const *shp, layout_mem_indirect const &rhs) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.chunk_total;
        return indirect_layout_mem::compare_indirect_blocks((void **)blocktab, (void **)rhs.blocktab, nblocks,
                                                            blocklen);
    }
    // find content hash of memory.
    //
    uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.chunk_total;
        return Tensor::content_hash_data_indirect(oldhash, (void **)blocktab, nblocks, blocklen, is_float);
    }
};
///////////////////////////////////////////
template <typename Linfo> class LayoutTensor;
template <typename Linfo> class BlockTableAccessor {
  protected:
    static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using pointer_type = storage_type *;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    static_assert(Linfo::is_indirect && Linfo::is_chunked);
    pointer_type *blktab; // the base of the block table
    std::array<size_t, Rank> blkdims; // dims of the block table in blocks
    std::array<size_t, Rank> blkstrides; // 'strides' (note stride for dim i is blkstrides[i+1];
    // stride for dim RANK-1  is 1; blkstrides[0] is the whole size.
    std::array<unsigned, Rank> margin; // margin offset
  public:
    explicit BlockTableAccessor(LayoutTensor<Linfo> const &tens) : blktab(tens.blocktab_ptr())
    {
        Shape<Rank> const &shp = *tens.shape;
        size_t allprod = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            unsigned const blk = TLayout::ChunkSizes[i];
            size_t const blkdim = shp.max_dims[i] / blk;
            allprod *= blkdim;
            blkdims[i] = blkdim;
            margin[i] = shp.pad[i];
            blkstrides[i] = allprod;
        }
    }
    // methods which have the same name as tensor methods, do
    // the same thing here.

    inline static constexpr unsigned rank() { return Rank; }

    inline size_t blocktab_len() const { return blkstrides[0]; }
    inline pointer_type *blocktab_ptr() const { return blktab; }
    inline size_t blocktab_blocksize() const { return TLayout::chunk_total; };
    inline size_t blocktab_blocksize_bytes() const { return TLayout::chunk_total * sizeof(storage_type); };

    inline size_t blocktab_dim(int i) const { return blkdims[i]; }
    inline size_t blocktab_dim_stride(int i) const { return (i < Rank - 1) ? blkstrides[i + 1] : 1; }

    // block_ptr_address(b,h,w,d) and block_ptr accept element coordinates.

    template <typename... ind_types> inline pointer_type *block_ptr_address(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return block_ptr_calc(coords);
    }
    template <typename... ind_types> inline pointer_type &block_ptr(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *block_ptr_calc(coords);
    }
    // blktab(b,h,w,d) accepts *block* coords
    //
    template <typename... ind_types> inline pointer_type &blocktab(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *blktab_ptr_calc(coords);
    }
    // same_table_shape: the shape of the table is the same as the 'other'.
    bool same_table_shape(BlockTableAccessor const &other) const
    {
        for (int i = 0; i < Rank; i++)
            if (blkdims[i] != other.blkdims[i]) return false;
        return true;
    }
    // 'same_layout' means the same table shape and the same padding offset. Dims may not be identical.
    bool same_layout(BlockTableAccessor const &other) const
    {
        if (!same_table_shape(other)) return false;
        for (int i = 0; i < Rank; i++)
            if (margin[i] != other.margin[i]) return false;
        return true;
    }

  protected:
    pointer_type *block_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = 0;
        for (int i = 0; i < Rank; i++) {
            unsigned blk = TLayout::ChunkSizes[i];
            unsigned idx = (coords[i] + margin[i] + (blk - 1)) / blk;
            sum += idx * ((i < Rank - 1) ? blkstrides[i + 1] : 1);
        }
        return blktab + sum;
    }
    pointer_type *blktab_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = coords[Rank - 1];
        for (int i = 0; i < Rank - 1; i++) {
            sum += coords[i] * blkstrides[i + 1];
        }
        return blktab + sum;
    }
};

//
// Constructors of LayoutTensor (all protected; only used by subclass ctor):
// LayoutTensor(const Op * producer_in, const OutputDef &def, Graph &graph_in, <<func pointer>>)
//    - build for given shape, attached to given producer.
//  LayoutTensor(const Op *producer_in, hnnx::Deserializer & dctx, <<funct pointer>>)
//    - deserialize. Notr that dctx contains a graph ref.
//  LayoutTensor(const ConcreteTensor &old, hnnx::Allocator *allocator,Tensor::clone_mode cmode)
//    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
//
// The function pointers in the first two cases are used to construct the correct
// interface object, according to the Interface_t of the subclass.
//

template <typename Linfo> class LayoutTensor : public RankedTensor<Linfo::Rank> {
  protected:
    using BaseRT = RankedTensor<Linfo::Rank>;
    static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    static constexpr bool is_chunked = Linfo::is_chunked;
    static_assert(is_chunked == (TLayout::chunk_total > 1));
    static constexpr bool is_indirect = Linfo::is_indirect;
    static constexpr bool is_padded = !std::is_same<Pad_t, NoPadding<Rank>>::value;

    static_assert(!(is_indirect && !is_chunked), "non-chunked layouts can't be indirect");

    Interface const *const interface_ptr; // pointer to shared instance of Interface subclass.
    using Shape_t = Shape<Rank>;

  public:
    Shape_t const *shape;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};

  protected: // interface, then shape, then mem
    using layout_mem_t = std::conditional_t<is_indirect, layout_mem_indirect<storage_type, TLayout, Pad_t>,
                                            layout_mem_contig<storage_type, TLayout, Pad_t>>;
    layout_mem_t mem;

  public:
    struct traits {
        using storage_type = LayoutTensor::storage_type;
        using raw_type = LayoutTensor::storage_type; // result from get_raw()
        static constexpr unsigned rank = Rank;
        static constexpr bool is_indirect = LayoutTensor::is_indirect;
        static constexpr bool is_chunked = LayoutTensor::is_chunked;
        static constexpr bool has_padding = !std::is_same<Pad_t, NoPadding<Rank>>::value;
        using pad_type = Pad_t;
        using layout_type = TLayout;
        using layouttensor_type = LayoutTensor;
    };

  protected:
    // ctors are marked noinline; otherwise they just get inlined
    // into all the ConcreteTensor ctors, which isn't really helpful.
    [[gnu::noinline]] LayoutTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in,
                                   Interface const *(*ifc_maker)(Graph &, OutputDef const &))
        : BaseRT(producer_in), interface_ptr((*ifc_maker)(graph_in, def)),
          shape(Shape_t::canonical_shape(
                  graph_in, Shape_t(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0]),
                                    mem.layout.pad(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0]))))),
          mem(shape, graph_in)
    {
    }
    [[gnu::noinline]] LayoutTensor(const Op *producer_in, hnnx::Deserializer &dctx,
                                   Interface const *(*ifc_deser_fp)(hnnx::Deserializer &))
        : BaseRT(producer_in, dctx), interface_ptr((*ifc_deser_fp)(dctx)), shape(Shape_t::deserialize(dctx)),
          mem(shape, dctx)
    {
    }
    // clone ctor.
    [[gnu::noinline]] LayoutTensor(const LayoutTensor &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode)
        : BaseRT(old, allocator, cmode), interface_ptr(old.interface_ptr), shape(old.shape),
          mem(shape, old.mem, allocator, cmode)
    {
    }

  public:
    virtual const inline size_t dim(size_t index) const noexcept override final { return shape->dims[index]; }
    const std::array<size_t, Rank> &dims() const { return shape->dims; };
    template <typename... T> const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        std::array<size_t, sizeof...(indices)> dim_sizes = {dim(indices)...};
        return dim_sizes;
    }
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept final
    {
        return std::pair<size_t const *, size_t>(&shape->dims[0], Rank);
    }
#if defined(NDEBUG) || defined(NO_SETDIMS_CHECK)
    virtual inline bool set_dims(const size_t dims[]) override final { return false; }
    virtual inline bool set_dims(const Tensor &prototype) override final { return false; }
#else
    virtual inline bool set_dims(const size_t dims[]) override
    {
        for (int i = 0; i < Rank; i++) {
            assert(dims[i] == shape->dims[i]);
        }
        return false;
    }
    virtual inline bool set_dims(const Tensor &prototype) override
    {
        auto [dims_p, dims_n] = prototype.get_dims();
        assert(dims_n == Rank);
        return set_dims(dims_p);
    }
#endif
    virtual Interface const &interface() const noexcept override { return *interface_ptr; }
    inline float interface_scale() const { return interface().get_scale(); }
    inline float interface_scale_recip() const { return interface().get_scale_recip(); }
    inline int32_t interface_offset() const { return interface().get_offset(); }

    // for direct access to bulk_data, in contiguous tensors only
    //  data_ptr() can be assigned to.
    inline std::conditional_t<is_indirect, void, storage_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        }
    }
    inline std::conditional_t<is_indirect, void, storage_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        }
    }

    // block table access
    inline storage_type **blocktab_ptr() const { return (storage_type **)mem.get_block_list_ptr(); }
    inline storage_type *&blocktab_at(size_t i)
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    inline storage_type *const &blocktab_at(size_t i) const
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    inline size_t blocktab_len() const { return mem.get_block_list_len(shape); }
    inline size_t blocktab_blocksize() const { return mem.get_elements_per_block(shape); }
    inline size_t blocktab_blocksize_bytes() const { return mem.get_elements_per_block(shape) * sizeof(storage_type); }

    // TODO: make total_storage elements have an optional bool parameter
    // to return in bytes; and then total_storage_bytes is a wrapper.
    virtual inline size_t total_storage_bytes() const final override
    {
        return total_storage_elements() * sizeof(storage_type);
    }
    virtual inline size_t total_storage_elements() const final override
    {
        size_t const total_elements =
                std::accumulate(shape->max_dims.cbegin(), shape->max_dims.cend(), 1, std::multiplies<size_t>());
        return total_elements;
    }
    virtual void *raw_data() noexcept override final { return mem.raw_data(); }
    virtual void set_raw_data_despite_danger(void *buffer) override final { mem.set_raw_data_despite_danger(buffer); }

    // change the padding; and reallocate blocktab if it's larger as a result.
    // in any case, all of the block pointers are zeroed.
    void change_pad(std::array<size_t, Rank> const &new_pad, hnnx::Allocator &allocator)
    {
        Shape_t newshape = *shape; // copy old shape
        for (int i = 0; i < Rank; i++)
            newshape.pad[i] = new_pad[i];
        newshape.max_dims = layout.pad(pad.pad_coords(newshape.dims, newshape.pad));
        // nake a persistent copy of new shape
        Shape_t const *const new_shape_p = Shape_t::canonical_shape(allocator.graph, newshape);
        // new_shape_p will be same pointer as shape, if shape wasn't changed. realloc_blocktab
        // checks for that.
        mem.realloc_blocktab(&allocator, shape, new_shape_p);
        shape = new_shape_p;
    }
    template <typename... ind_types> inline storage_type const *const *block_ptr_address(ind_types... inds) const
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> inline storage_type *const *block_ptr_address(ind_types... inds)
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> inline storage_type const *block_ptr(ind_types... inds) const
    {
        return *block_ptr_address(inds...);
    }
    template <typename... ind_types> inline storage_type *block_ptr(ind_types... inds)
    {
        return *block_ptr_address(inds...);
    }

    std::conditional_t<is_indirect, BlockTableAccessor<Linfo>, void> blocktable_accessor() const
    {
        if constexpr (is_indirect) {
            return BlockTableAccessor<Linfo>(*this);
        }
    }

    // this only makes sense for indirect tensors.
    std::conditional_t<is_indirect, std::array<size_t, Rank>, void> tile_strides() const
    {
        if constexpr (is_indirect) {
            std::array<size_t, Rank> ret = {0};
            ret[Rank - 1] = 1;
            for (int i = Rank - 2; i >= 0; i--) {
                ret[i] = ret[i + 1] * (shape->max_dims[i + 1] / layout.ChunkSizes[i + 1]);
            }
            return ret;
        }
    }

    // get_raw_addr(...) on this class gives a storage_type *.
    template <typename... ind_types> inline storage_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type const *)this->element_addr(Rank, coords.data());
    }
    template <typename... ind_types> inline storage_type *get_raw_addr(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type *)this->element_ptr(Rank, coords.data());
    }
    template <typename... ind_types> inline storage_type const &get_raw(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type const *)this->element_addr(Rank, coords.data());
    }
    template <typename... ind_types> inline storage_type &get_raw(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type *)this->element_addr(Rank, coords.data());
    }
    // tile interface. These are defined in tile_extract.h
    virtual void const *read_tile(unsigned flags, void *buffer, size_t b, int h, int w, int d) const override;
    virtual void write_tile(unsigned flags, void const *buffer, size_t b, int h, int w, int d) override;
    virtual unsigned tile_support_bits() const override;

    // Return a reference to *this; useful to get the layout base class reference
    // for any tensor class which has one.
    // So if you call func(in.layout_base(), out.layout_base()), where 'func'
    // is a template func, it will be specialized according to the layout of
    // in and out, not the subclass.
    inline LayoutTensor &layout_base() { return *this; }
    inline LayoutTensor const &layout_base() const { return *this; }

    // checksum for debug
    [[gnu::noinline]] virtual uint64_t get_checksum() const override
    {
        // NOLINTNEXTLINE(misc-const-correctness): Don't const this variable
        uint64_t chk = 0;
        if constexpr (Rank == 4) {
            auto [batch, heights, width, depth] = this->get_dims_4();
            // TODO : maybe add a special case for R4flat layout/no padding; one call to checksum_bytes.
            if (batch && heights && width && depth) {
                storage_type const x0 = *(storage_type const *)this->get_raw_addr(0, 0, 0, 0);
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < heights; h++) {
                        for (size_t w = 0; w < width; w++) {
                            for (size_t d = 0; d < depth; d++) {
                                storage_type x = *(storage_type const *)this->get_raw_addr(b, h, w, d);
                                x ^= x0;
                                union {
                                    storage_type as_x;
                                    uint8_t as_byte[sizeof(storage_type)];
                                } uu = {x};
                                chk = hnnx::checksum_bytes(chk, uu.as_byte, sizeof(storage_type));
                            }
                        }
                    }
                }
                chk ^= x0;
            }
        }
        return chk;
    }

  protected:
    // element_addr is delegated to the particular specialization of layout_mem
    virtual ALWAYSINLINE void *element_addr(size_t rank, const SIdx coords_in[]) const noexcept final override
    {
        return mem.element_addr(shape, rank, coords_in);
    }
    // compare_sametype is not overloaded here; LayoutTensor is an abstract class

    // This is called from ConcreteTensor::compare_sametype to fully compare two tensors
    // which are already known to be the same type (and have same interface)
    [[gnu::noinline]] int compare_sametype_layout(LayoutTensor const *rhs) const
    {
        if (shape->dims != rhs->shape->dims) return (shape->dims < rhs->shape->dims) ? -1 : 1;
        if (is_padded) {
            if (shape->max_dims != rhs->shape->max_dims) return (shape->max_dims < rhs->shape->max_dims) ? -1 : 1;
            // TODO: compare padding too. Maybe have a Padding method for this.
        }
        // compare memory now (delegate to layout_mem).
        return mem.compare_memory(shape, rhs->mem);
    }
    // allocation and enumeration.
    [[gnu::noinline]] void allocate_layout(hnnx::Allocator &allocator, unsigned options, MemoryClass mclass)
    {
        // get the pointer to block table; and number of entries in it.
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        size_t const blocksize = sizeof(storage_type) * this->mem.get_elements_per_block(this->shape);
        size_t const align = traits::is_indirect ? blocksize : std::min(size_t(256), sizeof(storage_type));

        allocator.allocate_n(blocktab, // pointer to pointers,
                             nblocks, // number of pointers
                             blocksize, align, mclass, options, this->get_dtype());
    }
    [[gnu::noinline]] void enum_memory_blocks_layout(hnnx::MemBlockEnumerator &en, MemoryClass mclass) const
    {
        // get the pointer to block table; and number of entries in it.
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        en.supply_blocks(this, mclass, (void *const *)blocktab, nblocks);
    }
    // called from find_content_hash in the ConcreteTensor class. hash_in includes
    // hash of dtype and interface.
    [[gnu::noinline]] uint32_t find_content_hash_layout(uint32_t hash_in, bool is_float) const noexcept
    {
        uint32_t h = hash_in ^ (Rank * 0x102401);
        h = Tensor::build_hash(shape->dims.data(), Rank, hash_in);
        if (is_padded) {
            h = Tensor::build_hash(shape->max_dims.data(), Rank, h);
            // TODO: including padding too (or instead)
        }
        return mem.find_content_hash(shape, h, is_float);
    }
    static const char *code_to_type_name() { return TensorTypeStruct<LayoutTensor<Linfo>>::name; }
};

//
// Constructors of ConcreteTensor:
// ConcreteTensor(const Op * producer_in, const OutputDef &def, Graph &graph_in)
//    - build for given shape, attached to given producer.
//  ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph & graph_in, T * data_in)
//    - same, but initialize pointer to given. Only available in 'flat' tensors.
//  ConcreteTensor(const Op *producer_in, hnnx::Deserializer & dctx)
//    - deserialize. Note that dctx contains a grap ref.
//  ConcreteTensor(const ConcreteTensor &old, hnnx::Allocator *allocator,Tensor::clone_mode cmode)
//    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
//

template <typename Tinfo> class ConcreteTensor : public LayoutTensor<typename Tinfo::Lconfig> {
  protected:
    using Interface_t = typename Tinfo::Interface_t;
    using Layout_t = typename Tinfo::Tlayout;
    using Pad_t = typename Tinfo::Pad_t;
    static constexpr DType dtype = dtype_of_type<Interface_t>();
    static constexpr bool is_indirect = Tinfo::is_indirect;
    static constexpr unsigned Rank = Layout_t::Rank;
    using BaseLayout = LayoutTensor<typename Tinfo::Lconfig>;
    using BaseRT = typename BaseLayout::BaseRT;

    // make sure it's compatible with supplied base class
    static_assert(Rank == BaseLayout::Rank && is_indirect == BaseLayout::traits::is_indirect &&
                          std::is_same<Layout_t, typename BaseLayout::traits::layout_type>::value &&
                          std::is_same<Pad_t, typename BaseLayout::traits::pad_type>::value,
                  "incompatible base class for ConcreteTensor");

  public:
    const char *true_name() const override { return Tinfo::typetag; };
    using Accessor_t = typename Interface_t::Accessor;
    using Const_Accessor_t = typename Interface_t::AccessorRO;
    using element_type = typename Interface_t::element_type;

    struct traits : public BaseLayout::traits {
        static constexpr DType dtype = ConcreteTensor::dtype;
        using element_type = typename dtype_traits<dtype>::element_type;
        using raw_type = element_type; // result from get_raw()
        using interface_type = Interface_t;
        static constexpr MemoryClass memclass = Tinfo::memclass;
    };
    //
    //  - build for given shape, attached to given producer.
    //  - pass the nase class ctor a specialized ctor, it uses to make the interface
    //   from the output def.
    ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
    }
    ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in, element_type *data_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
        this->mem.set_raw_data_despite_danger((void *)data_in);
    }
    //   - deserialize. Note that dctx contains a graph ref.
    //   We pass the base class a pointer to specialized function, which it uses to
    //  deserialize the interface.
    ConcreteTensor(const Op *producer_in, hnnx::Deserializer &dctx)
        : BaseLayout(producer_in, dctx, &hnnx::make_interface<Interface_t>::from_deser)
    {
    }
    //    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
    ConcreteTensor(const ConcreteTensor &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode)
        : BaseLayout(old, allocator, cmode)
    {
    }

    virtual DTypeScaleOff get_dtype_intfc() const noexcept override { return DTypeScaleOff(dtype, interface()); }

    // note: pp 10.3/5 of c++: I can override 'virtual Interface const &interface() const'
    // with 'virtual X const & interface() const;' if X is based on Interface. When this method
    // is called with a baser reference, the result is just quietly converted to Interface &.
    virtual Interface_t const &interface() const noexcept override final
    {
        return static_cast<Interface_t const &>(*this->interface_ptr);
    }
    inline float interface_scale() const { return interface().get_scale(); }
    inline float interface_scale_recip() const { return interface().get_scale_recip(); }
    inline int32_t interface_offset() const { return interface().get_offset(); }

    inline ALWAYSINLINE const element_type *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (element_type const *)this->element_addr(rank, coords);
    }
    inline ALWAYSINLINE element_type *element_ptr(size_t rank, const SIdx coords[])
    {
        return (element_type *)this->element_addr(rank, coords);
    }

    // Some methods return the same thing as in LayoutTensor, but
    // with the type being element_type instead of storage_type.
    inline std::conditional_t<is_indirect, void, element_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return (element_type *&)this->mem.bulk_data;
        }
    }
    inline std::conditional_t<is_indirect, void, element_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return (element_type *const &)this->mem.bulk_data;
        }
    }

    // block table access
    inline element_type **blocktab_ptr() const { return (element_type **)this->mem.get_block_list_ptr(); }
    inline element_type *&blocktab_at(size_t i) { return (element_type *&)BaseLayout::blocktab_at(i); }
    inline element_type *const &blocktab_at(size_t i) const
    {
        return (element_type *const &)BaseLayout::blocktab_at(i);
    }

    template <typename... ind_types> inline element_type const *const *block_ptr_address(ind_types... inds) const
    {
        return (element_type const *const *)BaseLayout::block_ptr_address(inds...);
    };
    template <typename... ind_types> inline element_type *const *block_ptr_address(ind_types... inds)
    {
        return (element_type *const *)BaseLayout::block_ptr_address(inds...);
    };
    template <typename... ind_types> inline element_type const *block_ptr(ind_types... inds) const
    {
        return *block_ptr_address(inds...);
    }
    template <typename... ind_types> inline element_type *block_ptr(ind_types... inds)
    {
        return *block_ptr_address(inds...);
    }

    // direct access methods.
    //
    template <typename... ind_types> inline Const_Accessor_t operator()(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {static_cast<SIdx>(inds)...};
        return Const_Accessor_t(this->element_addr(Rank, coords.data()), interface());
    }
    template <typename... ind_types> inline Accessor_t operator()(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return Accessor_t(this->element_addr(Rank, coords.data()), interface());
    }
    template <typename... ind_types> inline element_type const &get_raw(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type const *)this->element_addr(Rank, coords.data());
    }
    template <typename... ind_types> inline element_type &get_raw(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type *)this->element_addr(Rank, coords.data());
    }
    template <typename... ind_types> inline element_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type const *)this->element_addr(Rank, coords.data());
    }
    template <typename... ind_types> inline element_type *get_raw_addr(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type *)this->element_addr(Rank, coords.data());
    }
    virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return Tensor::formatcode_for_general<traits>();
    }
    virtual uint32_t get_tensor_info() const noexcept override
    {
        return Tensor::pack_tensor_info(traits::dtype, Rank, traits::memclass);
    }
    // allocation and enumeration.
    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override final
    {
        this->allocate_layout(allocator, options, traits::memclass);
    }
    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &en) const override
    {
        this->enum_memory_blocks_layout(en, traits::memclass);
    }
    // hash the dtype and interface, and let find_content_hash_layout do the rest.
    virtual uint32_t find_content_hash() const noexcept override final
    {
        uint32_t const h = interface().interface_hash() ^ mulu32_modular(unsigned(dtype), 0x107301);
        static constexpr bool is_float = dtype_traits<dtype>::is_float;
        return this->find_content_hash_layout(h, is_float);
    }

  protected:
    virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        // compare the interface, and then all the rest is done in compare_sametype_layout.
        auto *rhs = static_cast<const ConcreteTensor *>(rhs_in);
        int const icmp = interface().compare(rhs->interface());
        if (icmp != 0) return icmp;
        return this->compare_sametype_layout(rhs);
    }

    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              Tensor::tensor_blockinfo *infop) const override
    {
        void **retval = nullptr;
        ConcreteTensor const *newtens = nullptr;
        if (tensp) {
            *tensp = std::make_unique<ConcreteTensor>(*this, allocator, Tensor::clone_mode::duplicate);
            newtens = static_cast<ConcreteTensor const *>(tensp->get());
            retval = (void **)newtens->mem.get_block_list_ptr();
        }
        if (infop) {
            infop->setup(traits::dtype, traits::memclass);
            infop->blkptrs = (void **)this->mem.get_block_list_ptr();
            // pretend that a pointer to Shape<Rank> is really a pointer to its base class ShapeFlags
            // we provide a pointer to the shape field in the cloned tensor, if applicable; otherwise in 'this'.
            infop->shapepp = (hnnx::ShapeFlags *const *)&(newtens ? newtens : this)->shape;
            infop->nblocks = this->mem.get_block_list_len(this->shape);
            infop->blocksize = sizeof(element_type) * this->mem.get_elements_per_block(this->shape);
            infop->is_indirect = is_indirect;
            infop->is_chunked = traits::is_chunked;
            return retval;
        }
        return nullptr;
    }
    static const char *code_to_type_name() { return TensorTypeStruct<ConcreteTensor<Tinfo>>::name; }
};

template <typename T> class TensorIter;
template <typename T> class TensorCIter;

template <typename T> class IterableTensor {
    typedef TensorIter<T> iterator;
    typedef TensorCIter<T> const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;

  protected:
    pointer myTensor;
    const_pointer myCTensor;
    const std::array<size_t, 4> increments;
    mutable std::array<size_t, 4> dims;
    const bool is_const;

  public:
    friend iterator; //class TensorIter<T> ;
    friend const_iterator;

    inline IterableTensor(reference t, std::array<size_t, 4> inc)
        : myTensor(&t), myCTensor(const_cast<const_pointer>(&t)), increments(inc), is_const(false)
    {
        assert(myCTensor && myCTensor->rank() == 4);
        for (int i = 0; i < 4; i++) {
            dims[i] = myCTensor->dim(i);
        }
    }

    inline IterableTensor(const_reference t, std::array<size_t, 4> inc)
        : myTensor(nullptr), myCTensor(&t), increments(inc), is_const(true)
    {
        assert(myCTensor && myCTensor->rank() == 4);
        for (int i = 0; i < 4; i++) {
            dims[i] = myCTensor->dim(i);
        }
    }

    inline const size_t dim(size_t index) const { return dims[index]; }

    inline auto access(size_t b, size_t h, size_t w, size_t d) &
    {
        assert(!is_const && myTensor);
        return (*myTensor)(b, h, w, d);
    }
    inline auto access(size_t b, size_t h, size_t w, size_t d) &&
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }
    inline auto access(size_t b, size_t h, size_t w, size_t d) const &&
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    inline auto read(size_t b, size_t h, size_t w, size_t d) const
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    inline auto operator()(size_t b, size_t h, size_t w, size_t d) { return access(b, h, w, d); }
    inline const auto operator()(size_t b, size_t h, size_t w, size_t d) const
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    inline bool operator==(const IterableTensor<T> &it) const
    {
        return (this->myCTensor == it.myCTensor) && (this->increments == it.increments);
    }
    inline bool operator!=(const IterableTensor<T> &it) const { return !(*this == it); }

    inline iterator begin()
    {
        std::array<size_t, 4> const start = {0, 0, 0, 0};
        return iterator(*this, start);
    }
    inline const_iterator begin() const
    {
        std::array<size_t, 4> start = {0, 0, 0, 0};
        return const_iterator(*this, start);
    }
    inline iterator begin(std::array<size_t, 4> start) { return iterator(*this, start); }
    inline const_iterator begin(std::array<size_t, 4> start) const { return const_iterator(*this, start); }
    inline iterator end()
    {
        std::array<size_t, 4> const end = {dims[0], 0, 0, 0};
        return iterator(*this, end);
    }
    inline const_iterator end() const
    {
        std::array<size_t, 4> end = {dims[0], 0, 0, 0};
        return const_iterator(*this, end);
    }
    inline iterator end(std::array<size_t, 4> end) { return iterator(*this, end); }
    inline const_iterator end(std::array<size_t, 4> end) const { return const_iterator(*this, end); }

    ~IterableTensor() {}
};

template <typename T> class TensorIter {
  private:
    IterableTensor<T> &myITensor;
    std::array<size_t, 4> location;
    bool increment(size_t dim)
    {
        size_t const inc = myITensor.increments[dim];
        if (inc) {
            size_t const loc = location[dim];
            if (loc + inc < myITensor.dim(dim)) {
                location[dim] += inc;
                return true;
            } else if (dim != 0) {
                location[dim] = 0;
            }
        }
        if (dim == 0) {
            location[0]++;
            return true;
        }

        return false;
    }

    inline void incrementLocation()
    {
        int i = location.size();
        while (!increment(--i))
            ;
    }

  protected:
    inline TensorIter(const TensorIter<T> &to_copy) : myITensor(to_copy.myITensor), location(to_copy.location) {}

  public:
    inline TensorIter(IterableTensor<T> &it, std::array<size_t, 4> loc) : myITensor(it), location(loc) {}

    inline TensorIter<T> &clone() { return TensorIter<T>(*this); }

    inline std::array<size_t, 4> get_location() { return location; }

    inline bool operator==(const TensorIter<T> &ti) const
    {
        if (this->myITensor == ti.myITensor) {
            for (int i = 0; i < this->location.size(); i++) {
                if (this->location[i] != ti.location[i]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
    inline bool operator!=(const TensorIter<T> &ti) const { return !(*this == ti); }
    inline operator float() const { return myITensor(location[0], location[1], location[2], location[3]); }
    inline TensorIter<T> &operator=(const float v)
    {
        myITensor(location[0], location[1], location[2], location[3]) = v;
        return *this;
    }
    //inline TensorIter<T>&operator=(const TensorIter<T>& v) { return this->operator=(float(v)); }
    //inline auto & operator*() {return myITensor(location[0],location[1],location[2],location[3]);}
    inline TensorIter<T> &operator++()
    {
        incrementLocation();
        return *this;
    }
    inline TensorIter<T> operator++(int)
    {
        TensorIter<T> const clone = TensorIter<T>(*this);
        incrementLocation();
        return clone;
    }

    ~TensorIter() {}
};

template <typename T> class TensorCIter {
  private:
    const IterableTensor<T> &myITensor;
    std::array<size_t, 4> location;
    bool increment(size_t dim)
    {
        const size_t inc = myITensor.increments[dim];
        if (inc) {
            size_t const loc = location[dim];
            if (loc + inc < myITensor.dim(dim)) {
                location[dim] += inc;
                return true;
            } else if (dim != 0) {
                location[dim] = 0;
            }
        }
        if (dim == 0) {
            location[0]++;
            return true;
        }

        return false;
    }

    inline void incrementLocation()
    {
        int i = location.size();
        while (!increment(--i))
            ;
    }

  public:
    inline TensorCIter(const IterableTensor<T> &it, std::array<size_t, 4> loc) : myITensor(it), location(loc) {}

    inline std::array<size_t, 4> get_location() { return location; }

    inline bool operator==(const TensorCIter<T> &ti) const
    {
        if (this->myITensor == ti.myITensor) {
            for (int i = 0; i < this->location.size(); i++) {
                if (this->location[i] != ti.location[i]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
    inline bool operator!=(const TensorCIter<T> &ti) const { return !(*this == ti); }
    inline operator float() const { return myITensor.read(location[0], location[1], location[2], location[3]); }
    inline TensorCIter<T> &operator++()
    {
        incrementLocation();
        return *this;
    }
    inline TensorCIter<T> operator++(int)
    {
        TensorCIter<T> clone(*this);
        incrementLocation();
        return clone;
    }

    ~TensorCIter() {}
};
namespace Ldefs {
template <unsigned elbytes> struct stype_for;
template <> struct stype_for<1> {
    typedef uint8_t type;
};
template <> struct stype_for<2> {
    typedef uint16_t type;
};
template <> struct stype_for<4> {
    typedef NN_UINT32_T type;
};
} // namespace Ldefs
// macro to define a layout config struct in Ldefs namespace:
// paramaters are:
//  - name of struct (in Ldefs namespace)
//  - number of bytes per  storage element
//  - 'layout' type (which determines rank
//  - name of 'padding' template (Pading or NoPadding).
//
// Normally, if the layout is chunked, you get an indirect tensor.
// Use LAYOUTDEF_CONTIG to get a contiguous tensor with a chunked layout.
//
// Do not create different configurations with the same parameters;
// all this does is generate extra duplicate code.
//
#define LAYOUTDEF(NAME, ELBYTES, LAYOUT, PAD)                                                                          \
    namespace Ldefs {                                                                                                  \
    struct NAME {                                                                                                      \
        using Tlayout = LAYOUT;                                                                                        \
        using storage_type = stype_for<ELBYTES>::type;                                                                 \
        static constexpr unsigned Rank = Tlayout::Rank;                                                                \
        using Pad_t = PAD<Rank>;                                                                                       \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                                   \
        static constexpr bool is_indirect = is_chunked;                                                                \
    };                                                                                                                 \
    }
// define a layout config which has chunked addressing, but contiguous alloc.
#define LAYOUTDEF_CONTIG(NAME, ELBYTES, LAYOUT, PAD)                                                                   \
    namespace Ldefs {                                                                                                  \
    struct NAME {                                                                                                      \
        using Tlayout = LAYOUT;                                                                                        \
        using storage_type = stype_for<ELBYTES>::type;                                                                 \
        static constexpr unsigned Rank = Tlayout::Rank;                                                                \
        using Pad_t = PAD<Rank>;                                                                                       \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                                   \
        static constexpr bool is_indirect = false;                                                                     \
    };                                                                                                                 \
    }

#define DEFINE_TYPENAMES(TYPE, NAME)                                                                                   \
    DEFINE_TYPENAME(TYPE, NAME);                                                                                       \
    DEFINE_TYPENAME_V(Vector<const TYPE *>, NAME);

// Create function that accesses the TensorTypeStruct::name that places the map of opcode ->
// typename in .rodata.
// There are two versions of this function, one below (which is called specifically for those
// tensor types which are NOT one off : RankedTensor, TensorSclrDT, LayoutTensor,
// ConcreteTensor).
// If it is one of the above four tensor types, it is declared as a static member function which
// gets created during the explicity template specialisations below.
// Behaviour:
//  - If explicity specialised and one of RankedTensor, TensorSclrDT, LayoutTensor, ConcreteTensor,
//    static member function creates the map entry in .rodata.
//  - If not explicity specialised, need to call DECLARE_TENSOR_CODE_TO_TYPENAME_STRING macro in
//    order to place entry in .rodata
template <typename T> constexpr const char *code_to_type_name()
{
    return "unknown";
}

#define DECLARE_TENSOR_CODE_TO_TYPENAME_STRING(TYPE)                                                                   \
    template <> const char *code_to_type_name<TYPE>() { return TensorTypeStruct<TYPE>::name; }

// macro to define a ConcreteTensor config in Tdefs namespace
//  LAYOUTNAME is a layout defined by LAYOUTDEF macro
// DTYPE and MCLASS are just dtype and memory class.
// You must use a layout with element size matching the dtype.
//
// It is possible to create different configurations with
// the same paramaters; and in this way create different
// ConcreateTensor types which behave in the same way.
//
// For instamce, QFloatCrouton and Int32Crouton have different identities
// and the same configuration.
//
#define TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, MCLASS, ENCODENAME)                                                      \
    namespace Tdefs {                                                                                                  \
    struct NAME {                                                                                                      \
        using Lconfig = Ldefs::LAYOUTNAME;                                                                             \
        using Tlayout = Lconfig::Tlayout;                                                                              \
        using storage_type = Lconfig::storage_type;                                                                    \
        using element_type = dtype_traits<DTYPE>::element_type;                                                        \
        static_assert(sizeof(element_type) == sizeof(storage_type), "layout has wrong element size");                  \
        using Interface_t = std::conditional_t<dtype_traits<DTYPE>::is_quant, ScaleOffsetInterface<element_type>,      \
                                               PlainInterface<element_type>>;                                          \
        static constexpr size_t Rank = Lconfig::Rank;                                                                  \
        using Pad_t = Lconfig::Pad_t;                                                                                  \
        static constexpr bool is_chunked = Lconfig::is_chunked;                                                        \
        static constexpr bool is_indirect = Lconfig::is_indirect;                                                      \
        static constexpr MemoryClass memclass = MCLASS;                                                                \
        static constexpr const char *typetag = ENCODENAME;                                                             \
    };                                                                                                                 \
    }                                                                                                                  \
    DEFINE_TYPENAMES(ConcreteTensor<Tdefs::NAME>, ENCODENAME);

#define TENSORDEF(NAME, LAYOUTNAME, DTYPE, ENCODENAME)                                                                 \
    TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, MemoryClass::Default, ENCODENAME)

// LAYOUTDEF defines a configuration
//
LAYOUTDEF(Flat_8, 1, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_8, 1, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_16, 2, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_16, 2, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_32, 4, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_32, 4, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat6D_32, 4, R6FlatMemoryLayout, NoPadding)

LAYOUTDEF(Crouton_8, 1, R4CroutonLayout, Padding)
LAYOUTDEF(Crouton_16, 2, R4Crouton2Layout, Padding)
LAYOUTDEF(Crouton_32, 4, R4Crouton4Layout, Padding)
LAYOUTDEF(Crouton4x1_8, 1, R4Crouton4x1Layout, Padding)
LAYOUTDEF(Crouton2x2_8, 1, R4Crouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_8, 1, R4WideCroutonLayout, Padding)
LAYOUTDEF(WideCrouton2x2_8, 1, R4WideCrouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_32, 4, R4WideCrouton4Layout, Padding)

LAYOUTDEF(R4Depth32_32, 4, R4Depth32MemoryLayout, NoPadding)
LAYOUTDEF(R4Depth32_32pad, 4, R4Depth32MemoryLayout, Padding)

DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_8>, "yfB")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_8>, "yf5B")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_16>, "yfH")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_16>, "yf5H")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_32>, "yfI")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_32>, "yf5I")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat6D_32>, "yf6I")

DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_8>, "ycB")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_16>, "ycH")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_32>, "ycI")

// 5D LAYOUTDEFs for Croutons
// LAYOUTDEF(Crouton_8_5D, 1, R5CroutonLayout, Padding)
// LAYOUTDEF(Crouton_16_5D, 2, R5Crouton2Layout, Padding)
// LAYOUTDEF(Crouton_32_5D, 4, R5Crouton4Layout, Padding)

// TENSORDEF
// 8-bit
TENSORDEF(QuantUint8, Flat_8, DType::QUInt8, "fB")
TENSORDEF(QuantUint8_5D, Flat5D_8, DType::QUInt8, "f5B")
TENSORDEF(QuantInt8, Flat_8, DType::QInt8, "fb")
TENSORDEF(QuantInt8_5D, Flat5D_8, DType::QInt8, "f5b")
TENSORDEF(QUint8Crouton, Crouton_8, DType::QUInt8, "cB")
TENSORDEF(QUint8Crouton4x1, Crouton4x1_8, DType::QUInt8, "c#B")
TENSORDEF(QUint8Crouton2x2, Crouton2x2_8, DType::QUInt8, "c#B")
TENSORDEF(QUint8WideCrouton, WideCrouton_8, DType::QUInt8, "wB")
TENSORDEF(QUint8WideCrouton2x2, WideCrouton2x2_8, DType::QUInt8, "w#B")
TENSORDEF(QInt8Crouton, Crouton_8, DType::QInt8, "cb")

TENSORDEF_MC(QuantUint8_TCM, Flat_8, DType::QUInt8, MemoryClass::TCM, "FB")
TENSORDEF_MC(QuantUint8_5D_TCM, Flat5D_8, DType::QUInt8, MemoryClass::TCM, "F5B")
TENSORDEF_MC(QuantInt8_TCM, Flat_8, DType::QInt8, MemoryClass::TCM, "Fb")
TENSORDEF_MC(QuantInt8_5D_TCM, Flat5D_8, DType::QInt8, MemoryClass::TCM, "F5b")
TENSORDEF_MC(QUint8Crouton_TCM, Crouton_8, DType::QUInt8, MemoryClass::TCM, "CB")
TENSORDEF_MC(QUint8Crouton4x1_TCM, Crouton4x1_8, DType::QUInt8, MemoryClass::TCM, "C#B")
TENSORDEF_MC(QUint8Crouton2x2_TCM, Crouton2x2_8, DType::QUInt8, MemoryClass::TCM, "C#B")
TENSORDEF_MC(QUint8WideCrouton_TCM, WideCrouton_8, DType::QUInt8, MemoryClass::TCM, "WB")
TENSORDEF_MC(QUint8WideCrouton2x2_TCM, WideCrouton2x2_8, DType::QUInt8, MemoryClass::TCM, "W#B")
TENSORDEF_MC(QInt8Crouton_TCM, Crouton_8, DType::QInt8, MemoryClass::TCM, "Cb")

// 16-bit
TENSORDEF(QuantUint16, Flat_16, DType::QUInt16, "fH")
TENSORDEF(QuantUint16_5D, Flat5D_16, DType::QUInt16, "f5H")
TENSORDEF(QuantInt16, Flat_16, DType::QInt16, "fh")
TENSORDEF(QUint16Crouton, Crouton_16, DType::QUInt16, "cH")
TENSORDEF(QInt16Crouton, Crouton_16, DType::QInt16, "ch")
TENSORDEF(F16Crouton, Crouton_16, DType::Float16, "ce")
TENSORDEF(F16Weights, Flat_16, DType::Float16, "fw")
TENSORDEF(PlainFloat16, Flat_16, DType::Float16, "fe")
TENSORDEF(PlainFloat16_5D, Flat5D_16, DType::Float16, "f5e")

TENSORDEF_MC(QuantUint16_TCM, Flat_16, DType::QUInt16, MemoryClass::TCM, "FH")
TENSORDEF_MC(QuantUint16_5D_TCM, Flat5D_16, DType::QUInt16, MemoryClass::TCM, "F5H")
TENSORDEF_MC(QuantInt16_TCM, Flat_16, DType::QInt16, MemoryClass::TCM, "Fh")
TENSORDEF_MC(QUint16Crouton_TCM, Crouton_16, DType::QUInt16, MemoryClass::TCM, "CH")
TENSORDEF_MC(QInt16Crouton_TCM, Crouton_16, DType::QInt16, MemoryClass::TCM, "Ch")
TENSORDEF_MC(F16Crouton_TCM, Crouton_16, DType::Float16, MemoryClass::TCM, "Ce")
TENSORDEF_MC(F16Weights_TCM, Flat_16, DType::Float16, MemoryClass::TCM, "Fw")
TENSORDEF_MC(PlainFloat16_TCM, Flat_16, DType::Float16, MemoryClass::TCM, "Fe")
TENSORDEF_MC(PlainFloat16_5D_TCM, Flat5D_16, DType::Float16, MemoryClass::TCM, "F5e")

// 32-bit
TENSORDEF(Int32, Flat_32, DType::Int32, "fi")
TENSORDEF(Int32_5D, Flat5D_32, DType::Int32, "f5i")
TENSORDEF(Int32_6D, Flat6D_32, DType::Int32, "f6i")
TENSORDEF(QuantInt32, Flat_32, DType::QInt32, "fs")
TENSORDEF(PlainFloat, Flat_32, DType::Float32, "ff")
TENSORDEF(PlainFloat5D, Flat5D_32, DType::Float32, "f5f")
TENSORDEF(QFloat, Flat_32, DType::Int32, "ft")
TENSORDEF(D32Float, R4Depth32_32, DType::Float32, "rf")
TENSORDEF(D32PaddedFloat, R4Depth32_32pad, DType::Float32, "pf")
TENSORDEF(Int32Crouton, Crouton_32, DType::Int32, "ci")
TENSORDEF(QInt32Crouton, Crouton_32, DType::QInt32, "cs")
TENSORDEF(QInt32WideCrouton, WideCrouton_32, DType::QInt32, "ws")
TENSORDEF(QFloatCrouton, Crouton_32, DType::Int32, "ct")
TENSORDEF(FloatCrouton, Crouton_32, DType::Float32, "cf")

TENSORDEF_MC(Int32_TCM, Flat_32, DType::Int32, MemoryClass::TCM, "Fi")
TENSORDEF_MC(Int32_5D_TCM, Flat5D_32, DType::Int32, MemoryClass::TCM, "F5i")
TENSORDEF_MC(QInt32Crouton_TCM, Crouton_32, DType::QInt32, MemoryClass::TCM, "Cs")
TENSORDEF_MC(QInt32WideCrouton_TCM, WideCrouton_32, DType::QInt32, MemoryClass::TCM, "Ws")
TENSORDEF_MC(QuantInt32_TCM, Flat_32, DType::QInt32, MemoryClass::TCM, "Fs")
TENSORDEF_MC(PlainFloat_TCM, Flat_32, DType::Float32, MemoryClass::TCM, "Ff")
TENSORDEF_MC(PlainFloat_5D_TCM, Flat5D_32, DType::Float32, MemoryClass::TCM, "F5f")
TENSORDEF_MC(QFloat_TCM, Flat_32, DType::Int32, MemoryClass::TCM, "Ft")
TENSORDEF_MC(Int32Crouton_TCM, Crouton_32, DType::Int32, MemoryClass::TCM, "Ci")
TENSORDEF_MC(QFloatCrouton_TCM, Crouton_32, DType::Int32, MemoryClass::TCM, "Ct")
TENSORDEF_MC(FloatCrouton_TCM, Crouton_32, DType::Float32, MemoryClass::TCM, "Cf")

DEFINE_TYPENAMES(Vector<Tensor *>, "t*");
DEFINE_TYPENAMES(TensorScalar<float>, "nf");
DEFINE_TYPENAMES(TensorScalar<NN_INT32_T>, "ni");
DEFINE_TYPENAMES(TensorShape<1>, "s1");
DEFINE_TYPENAMES(TensorShape<2>, "s2");
DEFINE_TYPENAMES(TensorShape<3>, "s3");
DEFINE_TYPENAMES(TensorShape<4>, "s4");
DEFINE_TYPENAMES(TensorShape<5>, "s5");
DEFINE_TYPENAMES(Tensor, "t");

template <> constexpr const char *type_name<Graph>()
{
    return "";
}

extern template class ConcreteTensor<Tdefs::PlainFloat>;
extern template class ConcreteTensor<Tdefs::PlainFloat5D>;
extern template class ConcreteTensor<Tdefs::PlainFloat_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat_5D_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat16>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_5D>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QFloat>;
extern template class ConcreteTensor<Tdefs::QFloat_TCM>;
extern template class ConcreteTensor<Tdefs::D32Float>;
extern template class ConcreteTensor<Tdefs::D32PaddedFloat>;
extern template class ConcreteTensor<Tdefs::QuantUint8>;
extern template class ConcreteTensor<Tdefs::QuantUint8_5D>;
extern template class ConcreteTensor<Tdefs::QuantInt8>;
extern template class ConcreteTensor<Tdefs::QuantInt8_5D>;
extern template class ConcreteTensor<Tdefs::QuantUint16>;
extern template class ConcreteTensor<Tdefs::QuantUint16_5D>;
extern template class ConcreteTensor<Tdefs::QuantInt16>;
extern template class ConcreteTensor<Tdefs::QuantInt16_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt32>;
extern template class ConcreteTensor<Tdefs::QuantInt32_TCM>;
extern template class ConcreteTensor<Tdefs::Int32>;
extern template class ConcreteTensor<Tdefs::Int32_5D>;
extern template class ConcreteTensor<Tdefs::Int32_6D>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton>;
extern template class ConcreteTensor<Tdefs::QInt8Crouton>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QInt8Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton4x1>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton4x1_TCM>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton2x2>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton2x2_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint8_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint8_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint16_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt8_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt8_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::F16Crouton>;
extern template class ConcreteTensor<Tdefs::F16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::F16Weights>;
extern template class ConcreteTensor<Tdefs::F16Weights_TCM>;
extern template class ConcreteTensor<Tdefs::QInt32Crouton>;
extern template class ConcreteTensor<Tdefs::QInt32Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::Int32Crouton>;
extern template class ConcreteTensor<Tdefs::Int32Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QFloatCrouton>;
extern template class ConcreteTensor<Tdefs::QFloatCrouton_TCM>;
extern template class ConcreteTensor<Tdefs::FloatCrouton>;
extern template class ConcreteTensor<Tdefs::FloatCrouton_TCM>;

// standard layouts are instantiated in tensor.h
extern template class LayoutTensor<Ldefs::Flat_8>;
extern template class LayoutTensor<Ldefs::Flat_16>;
extern template class LayoutTensor<Ldefs::Flat_32>;
extern template class LayoutTensor<Ldefs::Flat5D_32>;
extern template class LayoutTensor<Ldefs::Flat6D_32>;

extern template class LayoutTensor<Ldefs::Crouton_8>;
extern template class LayoutTensor<Ldefs::Crouton_16>;
extern template class LayoutTensor<Ldefs::Crouton_32>;

template <typename T> // FIXME  - alias for transition
using TensorContiguous = ConcreteTensor<T>;

/////////////////////////
typedef ConcreteTensor<Tdefs::PlainFloat16_5D> PlainFloat16Tensor5D;
typedef ConcreteTensor<Tdefs::PlainFloat5D> PlainFloatTensor5D;
typedef ConcreteTensor<Tdefs::PlainFloat> PlainFloatTensor;
typedef ConcreteTensor<Tdefs::PlainFloat16> PlainFloat16Tensor;
typedef ConcreteTensor<Tdefs::D32Float> D32FloatTensor;
typedef ConcreteTensor<Tdefs::D32PaddedFloat> D32PaddedFloatTensor;
typedef ConcreteTensor<Tdefs::QuantUint8> QuantUint8Tensor;
typedef ConcreteTensor<Tdefs::QuantUint8_5D> QuantUint8Tensor5D;
typedef ConcreteTensor<Tdefs::QuantInt8> QuantInt8Tensor;
typedef ConcreteTensor<Tdefs::QuantInt8_5D> QuantInt8Tensor5D;
typedef ConcreteTensor<Tdefs::QuantUint16> QuantUint16Tensor;
typedef ConcreteTensor<Tdefs::QuantUint16_5D> QuantUint16Tensor5D;
typedef ConcreteTensor<Tdefs::QuantInt16> QuantInt16Tensor;
typedef ConcreteTensor<Tdefs::QuantInt32> QuantInt32Tensor;
typedef ConcreteTensor<Tdefs::Int32> Int32Tensor;
typedef ConcreteTensor<Tdefs::Int32_5D> Int32Tensor5D;
typedef ConcreteTensor<Tdefs::Int32_6D> Int32Tensor6D;
typedef ConcreteTensor<Tdefs::QUint8Crouton> QUint8CroutonTensor;
typedef ConcreteTensor<Tdefs::QInt8Crouton> QInt8CroutonTensor;
typedef ConcreteTensor<Tdefs::QUint16Crouton> QUint16CroutonTensor;
typedef ConcreteTensor<Tdefs::QInt16Crouton> QInt16CroutonTensor;
typedef ConcreteTensor<Tdefs::F16Crouton> F16CroutonTensor;
typedef ConcreteTensor<Tdefs::F16Weights> F16WeightsTensor;
typedef ConcreteTensor<Tdefs::QInt32Crouton> QInt32CroutonTensor;
typedef ConcreteTensor<Tdefs::Int32Crouton> Int32CroutonTensor;
typedef ConcreteTensor<Tdefs::QFloatCrouton> QFloatCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton> QUint8WideCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton2x2> QUint8WideCrouton2x2Tensor;
typedef ConcreteTensor<Tdefs::QInt32WideCrouton> QInt32WideCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8Crouton4x1> QUint8Crouton4x1Tensor;
typedef ConcreteTensor<Tdefs::QUint8Crouton2x2> QUint8Crouton2x2Tensor;
typedef ConcreteTensor<Tdefs::QFloat> QFloatTensor;

// These were once TensorContiguous
typedef ConcreteTensor<Tdefs::PlainFloat> PlainFloatContiguousTensor;
typedef ConcreteTensor<Tdefs::QFloat> QFloatContiguousTensor;

struct ModifiedDerivedTypeParent {
    using PlainFloatTensor_TCM = PlainFloatTensor;
    using PlainFloatTensor5D_TCM = PlainFloatTensor5D;
    using PlainFloat16Tensor_TCM = PlainFloat16Tensor;
    using PlainFloat16Tensor5D_TCM = PlainFloat16Tensor5D;
    using QFloatTensor_TCM = QFloatTensor;
    using QuantInt16Tensor_TCM = QuantInt16Tensor;
    using QuantInt32Tensor_TCM = QuantInt32Tensor;
    using QUint8CroutonTensor_TCM = QUint8CroutonTensor;
    using QInt8CroutonTensor_TCM = QInt8CroutonTensor;
    using QUint8Crouton4x1Tensor_TCM = QUint8Crouton4x1Tensor;
    using QUint8Crouton2x2Tensor_TCM = QUint8Crouton2x2Tensor;
    using QuantUint8Tensor_TCM = QuantUint8Tensor;
    using QuantUint8Tensor5D_TCM = QuantUint8Tensor5D;
    using QuantUint16Tensor_TCM = QuantUint16Tensor;
    using QuantUint16Tensor5D_TCM = QuantUint16Tensor5D;
    using QuantInt8Tensor_TCM = QuantInt8Tensor;
    using QuantInt8Tensor5D_TCM = QuantInt8Tensor5D;
    using QUint16CroutonTensor_TCM = QUint16CroutonTensor;
    using QInt16CroutonTensor_TCM = QInt16CroutonTensor;
    using F16CroutonTensor_TCM = F16CroutonTensor;
    using F16WeightsTensor_TCM = F16WeightsTensor;
    using QInt32CroutonTensor_TCM = QInt32CroutonTensor;
    using Int32CroutonTensor_TCM = Int32CroutonTensor;
    using QFloatCroutonTensor_TCM = QFloatCroutonTensor;
    using QUint8WideCroutonTensor_TCM = QUint8WideCroutonTensor;
    using QUint8WideCrouton2x2Tensor_TCM = QUint8WideCrouton2x2Tensor;
    using QInt32WideCroutonTensor_TCM = QInt32WideCroutonTensor;
    using Int32Tensor_TCM = Int32Tensor;
    using Int32Tensor5D_TCM = Int32Tensor5D;
};

/////////////////////////

typedef ConcreteTensor<Tdefs::PlainFloat_TCM> PlainFloatTensor_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat_5D_TCM> PlainFloatTensor5D_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat16_TCM> PlainFloat16Tensor_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat16_5D_TCM> PlainFloat16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QFloat_TCM> QFloatTensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt16_TCM> QuantInt16Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt32_TCM> QuantInt32Tensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton_TCM> QUint8CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QInt8Crouton_TCM> QInt8CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton4x1_TCM> QUint8Crouton4x1Tensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton2x2_TCM> QUint8Crouton2x2Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint8_TCM> QuantUint8Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint8_5D_TCM> QuantUint8Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QuantUint16_TCM> QuantUint16Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint16_5D_TCM> QuantUint16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QuantInt8_TCM> QuantInt8Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt8_5D_TCM> QuantInt8Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QUint16Crouton_TCM> QUint16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QInt16Crouton_TCM> QInt16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::F16Crouton_TCM> F16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::F16Weights_TCM> F16WeightsTensor_TCM;
typedef ConcreteTensor<Tdefs::QInt32Crouton_TCM> QInt32CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::Int32Crouton_TCM> Int32CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QFloatCrouton_TCM> QFloatCroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton_TCM> QUint8WideCroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton2x2_TCM> QUint8WideCrouton2x2Tensor_TCM;
typedef ConcreteTensor<Tdefs::QInt32WideCrouton_TCM> QInt32WideCroutonTensor_TCM;

// These were once TensorContiguous
typedef ConcreteTensor<Tdefs::Int32_TCM> Int32Tensor_TCM;
typedef ConcreteTensor<Tdefs::Int32_5D_TCM> Int32Tensor5D_TCM;
// typedef for layouts
typedef LayoutTensor<Ldefs::Flat_8> LayoutFlat_8;
typedef LayoutTensor<Ldefs::Flat5D_8> LayoutFlat5D_8;
typedef LayoutTensor<Ldefs::Flat_16> LayoutFlat_16;
typedef LayoutTensor<Ldefs::Flat5D_16> LayoutFlat5D_16;
typedef LayoutTensor<Ldefs::Flat_32> LayoutFlat_32;
typedef LayoutTensor<Ldefs::Flat5D_32> LayoutFlat5D_32;

// 'standard' crouton layouts.
typedef LayoutTensor<Ldefs::Crouton_8> LayoutCrouton_8; // [1,8,8,32]
typedef LayoutTensor<Ldefs::WideCrouton_8> LayoutWideCrouton_8; // [1,2,32,32]
typedef LayoutTensor<Ldefs::Crouton_16> LayoutCrouton_16; // [1,8,4,32] interleaved
typedef LayoutTensor<Ldefs::Crouton_32> LayoutCrouton_32; // [1,8,2,32]
typedef LayoutTensor<Ldefs::WideCrouton_32> LayoutWideCrouton_32; // [1,2,8,32]

typedef LayoutTensor<Ldefs::Crouton4x1_8> LayoutCrouton4x1_8;
typedef LayoutTensor<Ldefs::Crouton2x2_8> LayoutCrouton2x2_8;
typedef LayoutTensor<Ldefs::WideCrouton2x2_8> LayoutWideCrouton2x2_8;

using TypicalTensors =
        std::tuple<PlainFloatTensor, PlainFloatTensor5D, PlainFloat16Tensor, QuantUint8Tensor, QuantUint8Tensor5D,
                   QuantInt8Tensor, QuantInt8Tensor5D, QuantUint16Tensor, QuantUint16Tensor5D, QuantInt16Tensor,
                   QuantInt32Tensor, Int32Tensor, Int32Tensor5D, Int32Tensor6D, QUint8CroutonTensor, QInt8CroutonTensor,
                   QUint8Crouton4x1Tensor, QUint8Crouton2x2Tensor, QUint16CroutonTensor, QInt16CroutonTensor,
                   QInt32CroutonTensor, QFloatTensor, QFloatCroutonTensor, Int32CroutonTensor, PlainFloat16Tensor_TCM,
                   PlainFloat16Tensor5D>;

namespace hnnx {
// these tensor types are 'pre-registered' for deserialize

using CoreTensors =
        std::tuple<PlainFloatTensor, PlainFloatTensor5D, PlainFloat16Tensor, Int32Tensor, Int32Tensor5D, Int32Tensor6D,
                   PlainFloatTensor_TCM, PlainFloatTensor5D_TCM, Int32Tensor_TCM, QuantUint8Tensor, QuantUint8Tensor5D,
                   QuantInt8Tensor, QuantInt8Tensor5D, QuantUint8Tensor_TCM, QuantUint8Tensor5D_TCM,
                   QuantInt8Tensor_TCM, QuantInt8Tensor5D_TCM, QuantUint16Tensor, QuantUint16Tensor5D, QuantInt16Tensor,
                   QuantUint16Tensor_TCM, QuantUint16Tensor5D_TCM, QuantInt16Tensor_TCM, QuantInt32Tensor,
                   QUint8CroutonTensor, QuantInt32Tensor_TCM, QUint8CroutonTensor_TCM, QInt8CroutonTensor,
                   QUint16CroutonTensor, QInt8CroutonTensor_TCM, QUint16CroutonTensor_TCM, QInt32CroutonTensor,
                   QInt16CroutonTensor, QInt16CroutonTensor_TCM, QInt32CroutonTensor_TCM, QInt32WideCroutonTensor,
                   QInt32WideCroutonTensor_TCM, QFloatTensor, QFloatCroutonTensor, Int32CroutonTensor,
                   Int32CroutonTensor_TCM, D32FloatTensor, D32PaddedFloatTensor, F16CroutonTensor, F16CroutonTensor_TCM,
                   QUint8WideCroutonTensor, QUint8WideCroutonTensor_TCM, QUint8Crouton2x2Tensor_TCM,
                   QUint8WideCrouton2x2Tensor_TCM, PlainFloat16Tensor_TCM, PlainFloat16Tensor5D, Int32Tensor5D_TCM>;

const char *get_op_true_name(const Op *op);

////// Tensor Generator //////////////

template <typename T, typename TX>
inline std::unique_ptr<T> make_tensor_template(Op const *op, OutputDef const &odef, Graph &g)
{
    return std::unique_ptr<T>(std::make_unique<TX>(op, odef, g));
}

// we make tables of these entries:
//  rank, dtype, pointer to function which makes it.
// The tables are built only as static constexpr variable in tensor_generator_lookup<T>::lookup
// so there should be only one table per TensorType after link.
//
struct tensor_generator_table_entry {
    typedef Tensor T; // maybe needs to be a template parm
    typedef std::unique_ptr<T> (*maketens_funcp)(Op const *, OutputDef const &, Graph &);

    int rank;
    DType dtype;
    maketens_funcp fp;

    // default ctor
    inline constexpr tensor_generator_table_entry() : rank(), dtype(), fp() {}

    // each entry is constructed based on pointer to the tensor type.
    template <typename TX>
    inline constexpr tensor_generator_table_entry(TX const *)
        : rank(tensor_traits<TX>::rank), dtype(tensor_traits<TX>::dtype), fp(make_tensor_template<T, TX>)
    {
    }
};
// a thing to make the constexpr table..
template <typename TTUPLE, size_t... I>
inline constexpr std::array<tensor_generator_table_entry, std::tuple_size_v<TTUPLE>>
        make_tengen_init(std::index_sequence<I...>)
{
    return {tensor_generator_table_entry(static_cast<typename std::tuple_element_t<I, TTUPLE> *>(nullptr))...};
}

template <typename TensorType> struct tensor_generator_lookup {
    template <typename TX>
    using has_TensorType_as_base = std::integral_constant<bool, std::is_base_of<TensorType, TX>::value>;
    // this is a tuple of types for which T is a common base.
    using applicable_types = TupFilter_t<has_TensorType_as_base, TypicalTensors>;
    static constexpr size_t NTYPES = std::tuple_size_v<applicable_types>;

    static tensor_generator_table_entry const *lookup(int rank, DType dtype)
    {
        // this is a table of their rank, dtype, ctor function.
        static constexpr std::array<tensor_generator_table_entry, NTYPES> typedescs =
                make_tengen_init<applicable_types>(std::make_index_sequence<NTYPES>{});
        tensor_generator_table_entry const *p = typedescs.data();

        for (int i = 0; i < int(NTYPES); i++) {
            if (p->dtype == dtype && p->rank == rank) return p;
            p++;
        }
        return nullptr;
    }

    static std::unique_ptr<Tensor> make [[gnu::noinline]] (const Op *producer_in, const OutputDef &def, Graph &graph_in)
    {
        // concrete types get a shortcut if the dtype & rank match...
        if constexpr (!std::is_abstract<TensorType>::value) {
            if (def.dtype == tensor_traits<TensorType>::dtype && def.rank == tensor_traits<TensorType>::rank) {
                return make_tensor_template<Tensor, TensorType>(producer_in, def, graph_in);
            }
        }
        tensor_generator_table_entry const *const lookup_result = lookup(def.rank, def.dtype);
        if (lookup_result != nullptr) {
            return lookup_result->fp(producer_in, def, graph_in);
        }
        errlog("Lookup in %d tensor types failed (%p: <<%s>>)", int(NTYPES), producer_in,
               get_op_true_name(producer_in));
        return nullptr;
    }
    // return true if 'make' would succeed.
    static bool is_valid(const OutputDef &def)
    {
        if constexpr (!std::is_abstract<TensorType>::value) {
            if (def.dtype == tensor_traits<TensorType>::dtype && def.rank == tensor_traits<TensorType>::rank)
                return true;
            else {
                debuglog(
                        "def.dtype %u, tensor_traits<TensorType>::dtype %u, def.rank %u, tensor_traits<TensorType>::rank %u",
                        def.dtype, tensor_traits<TensorType>::dtype, def.rank,
                        unsigned(tensor_traits<TensorType>::rank));
            }
        }
        return lookup(def.rank, def.dtype) != nullptr;
    }
};
// external API of tensor generator:
//    tensor_generator<T>( Op const *, OutputDef const &, Graph &) ->  std::unique_ptr<Tensor>
//    tensor_generator_valid<T>( Op const *, OutputDef const &, Graph &) ->  bool
//
// A call to tensor_generator<T>(..) is really a call to tensor_generator_lookup<T>::make(..)
//

bool tensor_tall_crouton_disabled(Graph const &g);
bool tensor_wide_crouton_disabled(Graph const &g);

template <typename T, typename = void> struct is_wide_crouton {
    static constexpr bool value = false;
};
template <typename T, typename = void> struct is_tall_crouton {
    static constexpr bool value = false;
};
template <typename T> struct is_wide_crouton<T, std::void_t<decltype(T::layout)>> {
    static constexpr bool value = (T::layout.chunk_total == 8 * 8 * 32) && (T::layout.ChunkSizes[2] > 1) &&
                                  (T::layout.ChunkSizes[1] < T::layout.ChunkSizes[2]);
};
template <typename T> struct is_tall_crouton<T, std::void_t<decltype(T::layout)>> {
    static constexpr bool value = (T::layout.chunk_total == 8 * 8 * 32) && (T::layout.ChunkSizes[1] > 1) &&
                                  (T::layout.ChunkSizes[1] >= T::layout.ChunkSizes[2]);
};

template <typename TensorType>
constexpr std::unique_ptr<Tensor> (*tensor_generator)(const Op *producer_in, const OutputDef &def,
                                                      Graph &graph_in) = tensor_generator_lookup<TensorType>::make;
template <typename TensorType>
inline bool tensor_generator_valid(const Op *producer_in, const OutputDef &def, Graph &graph_in)
{
    if constexpr (is_wide_crouton<TensorType>::value) {
        if (tensor_wide_crouton_disabled(graph_in)) {
            debuglog("Wide croutons disabled...");
            return false;
        }
    }
    if constexpr (is_tall_crouton<TensorType>::value) {
        if (tensor_tall_crouton_disabled(graph_in)) {
            debuglog("Tall croutons disabled...");
            return false;
        }
    }
    return tensor_generator_lookup<TensorType>::is_valid(def);
}

// make a scalar tensor for a given def (with 0 rank, and specific dtype). Returns an empty
// pointer if there is no support.
std::unique_ptr<Tensor> tensor_generator_scalar(const Op *producer_in, const OutputDef &def, void const *data,
                                                size_t len);

template <int relative_tolerance = 1 /* 1% */, int absolute_tolerance = 1 /* in 'FLT_EPSILON' ref <climits> */>
static inline constexpr int almost_eq(float rhs, float lhs)
{
    return std::abs(rhs - lhs) <= (
                                          // should it be max of (absolute, relative) ?
                                          (absolute_tolerance * std::numeric_limits<float>::epsilon()) +
                                          (relative_tolerance / 100.0 * std::abs(lhs)));
}

using cmp_function = std::function<int(float, float)>;
extern GraphStatus tensor_compare(const Tensor &lhs, const Tensor &rhs, cmp_function fn);
extern GraphStatus tensor_copy(Tensor &lhs, const Tensor &rhs);
extern GraphStatus check_dims(const Tensor &lhs, const Tensor &rhs);

//
// Set the shape of Tensor D to the same as Tensor S, and
// then copy the contents, adapting to whatever shapes and data format
//
void tensor_copy_4d(Tensor &dst, Tensor const &src);

void tensor_registry_testing();

template <typename T> struct memclass_of {
    static constexpr MemoryClass memclass = tensor_traits<T>::memclass;
};
template <> struct memclass_of<Tensor> {
    static constexpr MemoryClass memclass = MemoryClass::Default;
};

template <typename T> struct memclass_of<Vector<T *>> {
    static constexpr MemoryClass memclass = memclass_of<T>::memclass;
};
template <typename T> struct memclass_of<const Vector<T *>> {
    static constexpr MemoryClass memclass = memclass_of<T>::memclass;
};

template <MemoryClass C, typename... Ts> struct has_memclass;

template <MemoryClass C> struct has_memclass<C, std::tuple<>> {
    static constexpr bool value = false;
};

template <MemoryClass C, typename T, typename... Ts> struct has_memclass<C, std::tuple<T, Ts...>> {
    static constexpr bool value = memclass_of<T>::memclass == C || has_memclass<C, std::tuple<Ts...>>::value;
};

///////////////////////////////////////

// mechanism to generate functions to register tensor for serializing
// reg_tens_for_deser<T1,...>::f() -> int is a static function which
// registers T1,T2.
// To keep code size down, the code is only there for <T>; for more than
// one, the others are all called in sequence.
//
//  reg_tens_for_deser<T1,...>::f_ptr() is a static inline which returns
//  a pointer to f.
//
template <typename... T> struct reg_tens_for_deser {
    static int f() { return (reg_tens_for_deser<T>::f(), ...); }
    static constexpr auto f_ptr() -> int (*)() { return &f; }
};
// For empty list make fptr return null
// since most of the Ops have nothing to do, and a null pointer takes
// less code to make than the address of a function.
template <> struct reg_tens_for_deser<> {
    static constexpr auto f_ptr() -> int (*)() { return nullptr; }
};

// single item
//
template <typename T> struct reg_tens_for_deser<T> {
    static int f()
    {
        using TT = std::remove_reference_t<std::remove_cv_t<T>>;
        static_assert(std::is_same_v<T, TT>);
        if constexpr (!(std::is_abstract<T>::value)) {
            deserialize_tensor_register(typeid(T), type_name<T>(),
                                        deserialize_tensor_using_constructor<T>::deserialize);
        }
        return 0;
    }
    static constexpr auto f_ptr() -> int (*)()
    {
        if constexpr (!(std::is_abstract<T>::value)) {
            return &f;
        } else {
            // let's just have one empty function
            return reg_tens_for_deser<>::f_ptr();
        }
    }
};

template <typename TUP> struct map_rtfd_type {
};
template <typename... T> struct map_rtfd_type<std::tuple<T...>> {
    using type = reg_tens_for_deser<T...>;
};
// given a tuple TUP, deserialize_tensor_tuple<TUP,FORCE>::f_ptr()
// returns a pointer to a function
// which registers all of the types which are not in SkipRegTensors<FORCE>
// (i.e. if FORCE is false, all of the types which are not in CoreTensors;
// if force is true, all of the types).
//

// This is SkipRegTensors<True>, which is used for FORCE=true (don't skip).
// SkipRegTensors<false> is defined at the bottom of tensors.h.
template <bool FORCE> struct SkipRegTensors {
    using type = std::tuple<>;
};

template <typename TUP, bool FORCE = false> struct deserialize_tensor_tuple {
    template <typename T> using not_core_tensor = not_contains_type<typename SkipRegTensors<FORCE>::type, T>;
    using filtered_T = std::conditional_t<FORCE, TUP, typename TupFilter<not_core_tensor, TUP>::type>;
    using rtfd_type = typename map_rtfd_type<filtered_T>::type;
    static constexpr auto f_ptr() -> int (*)() { return rtfd_type::f_ptr(); }
    static int f() { return rtfd_type::f(); }
};

template <> struct SkipRegTensors<false> {
    using type = CoreTensors;
};

// map a tensor type to its layout tensor.
template <typename TT> using layout_of = typename tensor_traits<TT>::layouttensor_type;

} // namespace hnnx

POP_VISIBILITY()

#include "tile_extract.h"
#endif
