//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#ifndef OP_INFO_H
#define OP_INFO_H

#include <typeinfo>
#include <typeindex>
#include <functional>
#include <map>

#include "flags.h"
#include "op_registry.h"
#include "cost_funcs.h"
#include "weak_linkage.h"

PUSH_VISIBILITY(default)

class Op;

namespace hnnx {

class OpInfo {
    cost_function_t cost;
    Flags_word flags;
    bool is_external_flag;
    bool is_simple_op;
    union {
        OpFactory op_factory;
        SimpleOpFactory simple_op_factory;
    };
    const std::string_view type_tag;

  public:
    OpInfo(cost_function_t cost_in, Flags_word flags_in, OpFactory op_factory_in, bool is_external_in,
           const std::string_view type_tag_in)
        : cost(cost_in), flags(flags_in), is_external_flag(is_external_in), is_simple_op(false),
          op_factory(op_factory_in), type_tag(type_tag_in)
    {
    }
    OpInfo(cost_function_t cost_in, Flags_word flags_in, SimpleOpFactory simple_op_factory_in, bool is_external_in,
           const std::string_view type_tag_in)
        : cost(cost_in), flags(flags_in), is_external_flag(is_external_in), is_simple_op(true),
          simple_op_factory(simple_op_factory_in), type_tag(type_tag_in)
    {
    }

    ~OpInfo() = default;

    Flags_word get_flags() const { return flags; }

    cost_function_t const &get_cost() const { return cost; }

    bool is_external() const { return is_external_flag; }

    const char *get_type_tag() const { return type_tag.data(); }

    OpFactory get_op_factory() const { return !is_simple_op ? op_factory : nullptr; }
    SimpleOpFactory get_simple_op_factory() const { return is_simple_op ? simple_op_factory : nullptr; }
};

using InfoMapType = std::map<std::type_index, OpInfo>;

// after the instance is constructed, this points to it.
extern InfoMapType *op_info_map_inst_p;
InfoMapType &get_op_info_map_function();

inline InfoMapType &get_op_info_map()
{
    return (op_info_map_inst_p != nullptr) ? *op_info_map_inst_p : get_op_info_map_function();
}

// most access to the map are lookup. This does a lookup and returns null if not found.
OpInfo const *op_info_map_lookup(std::type_index tind);

// handy adapters
inline OpInfo const *op_info_map_lookup(std::type_info const &t)
{
    return op_info_map_lookup(std::type_index(t));
}
template <typename OP> // can't just use Op since it's incomplete here.
inline OpInfo const *op_info_map_lookup(OP const *op)
{
    static_assert(std::is_base_of<Op, OP>::value);
    return op_info_map_lookup(std::type_index(typeid(*op)));
}

void register_op_info(const std::type_info &type, cost_function_t cost, Flags_word flags, OpFactory op_factory,
                      bool is_external, const std::string_view type_tag);
void register_op_info(const std::type_info &type, cost_function_t cost, Flags_word flags, SimpleOpFactory op_factory,
                      bool is_external, const std::string_view type_tag);

template <typename T, typename OPFACTORY>
inline void register_op_info(cost_function_t cost, Flags_word flags, OPFACTORY op_factory, bool is_external,
                             const std::string_view type_tag)
{
    return register_op_info(typeid(T), cost, flags, op_factory, is_external, type_tag);
}

} // namespace hnnx

POP_VISIBILITY()

#endif
