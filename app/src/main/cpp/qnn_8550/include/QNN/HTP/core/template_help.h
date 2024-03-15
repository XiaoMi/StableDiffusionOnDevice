//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef HEXNN_TEMPLATE_HELP_H
#define HEXNN_TEMPLATE_HELP_H 1

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include "type_name.h"

class Graph;
template <typename P> class Vector;

namespace hnnx {

/* Wrap Types or Values in Templates */
/* I'm not sure that these are always needed, but it came in handy as I'm learning these things */

template <template <typename> typename Tname> struct TemplateTypeWrapper {
};

template <template <size_t> typename Tname> struct TemplateIdxWrapper {
};

template <typename twrap, size_t val> struct UnwrapIdxTemplate_struct {
};

template <template <size_t> typename Twrap, size_t Val>
struct UnwrapIdxTemplate_struct<TemplateIdxWrapper<Twrap>, Val> {
    using type = Twrap<Val>;
};

template <typename Twrap, size_t Val> using UnwrapIdxTemplate = typename UnwrapIdxTemplate_struct<Twrap, Val>::type;

template <typename twrap, typename tapply> struct UnwrapTypeTemplate_struct {
};

template <template <typename> typename Twrap, typename Tapply>
struct UnwrapTypeTemplate_struct<TemplateTypeWrapper<Twrap>, Tapply> {
    using type = Twrap<Tapply>;
};

template <typename Twrap, typename Tapply>
using UnwrapTypeTemplate = typename UnwrapTypeTemplate_struct<Twrap, Tapply>::type;

/*
 * Helper functions for dealing with tuples.
 * 
 * FIXME: EJP: some of these things might need to be refactored, as sometimes
 * they have some extra functionality for some specific use, instead of being as 
 * generic and reusable as possible...
 *
 */

/*
 * EJP: As I'm getting better at all this template stuff,
 * I should go back and refactor all this TypeFilter stuff.
 */

template <typename...> struct TupleCons;

/*
 * Create a tuple type of one element and the contents of an additional tuple
 */
template <template <typename...> typename C, typename T, typename... Rest> struct TupleCons<T, C<Rest...>> {
    using type = C<T, Rest...>;
};

template <template <typename> class Pred, template <typename> class Wrapper, typename...> struct TypeFilter;

/*
 * Just a single element: create empty tuple or tuple with the single element
 */
template <template <typename> class Pred, template <typename> class Wrapper, typename Head>
struct TypeFilter<Pred, Wrapper, Head> {
    using type =
            std::conditional_t<Pred<std::remove_reference_t<std::remove_pointer_t<Head>>>::value,
                               std::tuple<Wrapper<std::remove_reference_t<Head>>>, // FIXME: remove remove_reference_t
                               std::tuple<>>;
};
/*
 * Filter this element and concatenate with the rest of the elements
 */

/*
 * EJP: Maybe change this to take a tuple, so that we can refine
 */
template <template <typename> class Pred, template <typename> class Wrapper, typename Head, typename... Tail>
struct TypeFilter<Pred, Wrapper, Head, Tail...> {
    using type = std::conditional_t<Pred<std::remove_reference_t<std::remove_pointer_t<Head>>>::value,
                                    // FIXME: remove remove_reference_t here...
                                    typename TupleCons<Wrapper<std::remove_reference_t<Head>>,
                                                       typename TypeFilter<Pred, Wrapper, Tail...>::type>::type,
                                    typename TypeFilter<Pred, Wrapper, Tail...>::type>;
};

template <template <typename> typename Pred, typename...> struct TupFilter;
//template<template<typename> class Pred, template<typename> class Wrapper, typename...> struct TupFilter;

//template<template<typename> class Pred, template<typename> class Pred2, typename Head>
//template<template<typename...> typename C, template<typename> class Pred, typename Head>

template <template <typename> typename Pred, template <typename...> typename C> struct TupFilter<Pred, C<>> {
    using type = C<>;
};

template <template <typename> typename Pred, template <typename...> typename C, typename Head>
struct TupFilter<Pred, C<Head>> {
    using type = std::conditional_t<Pred<Head>::value, C<Head>, C<>>;
};

template <template <typename> typename Pred, template <typename...> typename C, typename Head, typename... Rest>
struct TupFilter<Pred, C<Head, Rest...>> {
    using type = std::conditional_t<Pred<Head>::value,
                                    typename TupleCons<Head, typename TupFilter<Pred, C<Rest...>>::type>::type,
                                    typename TupFilter<Pred, C<Rest...>>::type>;
};

template <template <typename> class Wrap, typename...> struct TupMap;

template <template <typename> typename Wrap, template <typename...> typename C> struct TupMap<Wrap, C<>> {
    using type = C<>;
};

template <template <typename> class Wrap, template <typename...> typename C, typename... Rest>
struct TupMap<Wrap, C<Rest...>> {
    using type = C<Wrap<Rest>...>;
};

#if 0
template <template <typename> class Wrap, template <typename...> typename C,
          typename... Ts>
using TupMap_t = typename TupMap<Wrap, C<Ts...>>::type;

template <template <typename> class Filt, template <typename...> typename C,
          typename... Ts>
using TupFilter_t = typename TupFilter<Filt, C<Ts...>>::type;
#else
template <template <typename> class Wrap, typename Tup> using TupMap_t = typename TupMap<Wrap, Tup>::type;

template <template <typename> class Filt, typename Tup> using TupFilter_t = typename TupFilter<Filt, Tup>::type;
#endif

template <typename T> struct Unboxed {
    using type = T;
};

template <typename T> struct Unboxed<const Vector<T>> {
    using type = T;
};
template <typename T> struct Unboxed<Vector<T>> {
    using type = T;
};

//template<template<typename...> typename C, typename T, typename... Ts>
//struct Unboxed<const C<T,Ts...>> {
//	using type = T;
//};

template <typename T> using unboxed_t = typename Unboxed<T>::type;

template <class T>
using is_not_const = std::integral_constant<bool, !std::is_const<std::remove_pointer_t<unboxed_t<T>>>::value>;
template <class T>
using is_const = std::integral_constant<bool, std::is_const<std::remove_pointer_t<unboxed_t<T>>>::value>;

//template<template<typename...> typename C, typename...>
template <typename T, typename Default> struct First_Tuple_Element {
};

template <template <typename...> typename C, typename First, typename... Rest, typename Default>
struct First_Tuple_Element<C<First, Rest...>, Default> {
    using type = First;
};

template <template <typename...> typename C, typename Default> struct First_Tuple_Element<C<>, Default> {
    using type = Default;
};

template <typename T, typename Default> using first_tuple_element = typename First_Tuple_Element<T, Default>::type;

/*
 * Use index sequence to turn a normal pointer unknown size array into a fixed size std::array
 *
 * Maybe this could get refactored into some kind of thing like tuple map
 */

template <size_t N, typename T, size_t... I>
constexpr static inline const std::array<T, N> ptr_to_stdarray_helper(const T *carray, std::index_sequence<I...>)
{
    const std::array<T, N> ret = {{carray[I]...}};
    return ret;
}

template <size_t N, typename T> constexpr static inline const std::array<T, N> ptr_to_stdarray(const T *carray)
{
    return ptr_to_stdarray_helper<N, T>(carray, std::make_index_sequence<N>{});
}

template <typename T> using is_boxed_val = std::integral_constant<bool, !std::is_same<T, unboxed_t<T>>::value>;

template <typename T> using is_unboxed_val = std::integral_constant<bool, std::is_same<T, unboxed_t<T>>::value>;

template <typename T> using is_boxed_const = std::integral_constant<bool, is_const<unboxed_t<T>>::value>;

template <typename T> using is_boxed_nonconst = std::integral_constant<bool, is_not_const<unboxed_t<T>>::value>;

template <typename T> using is_const_graph = std::integral_constant<bool, std::is_same<const Graph, T>::value>;

template <typename T> using is_not_const_graph = std::integral_constant<bool, !std::is_same<const Graph, T>::value>;

/*
 * These are kind of like add_pointer / add_pointer_t
 */

template <typename T> struct add_uniqueptr {
    using type = typename std::unique_ptr<T>;
};
template <class T> using add_uniqueptr_t = typename add_uniqueptr<T>::type;

template <class T> using add_nothing_t = T;

template <typename R> struct ArgsTuples;

template <typename R, typename... Args> struct ArgsTuples<R(Args...)> {
    static_assert((((std::is_lvalue_reference<Args>::value) || (std::is_rvalue_reference<Args>::value)) && ...),
                  "Op Arguments should be by reference");
    using all_args_tuple = std::tuple<Args...>; // all the args
    using noref_args_tup = TupMap_t<std::remove_reference_t,
                                    all_args_tuple>; // without references
    using const_graph_tup = TupFilter_t<is_const_graph, noref_args_tup>; // reference to graph?
    using nonconst_graph_tup = TupFilter_t<is_not_const_graph,
                                           noref_args_tup>; // not reference to graph
    using fixed_args = TupFilter_t<is_unboxed_val, nonconst_graph_tup>; // non-varargs
    using var_args = TupFilter_t<is_boxed_val, noref_args_tup>; // varargs

    using input_tuple = TupFilter_t<is_const, fixed_args>; // the inputs as real types
    using output_tuple = TupFilter_t<is_not_const, fixed_args>; // the outputs as real types
    using input_ptr_tuple = TupMap_t<std::add_pointer_t, input_tuple>; // The inputs as pointers
    using output_ptr_tuple = TupMap_t<std::add_pointer_t,
                                      output_tuple>; // the outputs as pointers
    using output_uniqueptrs_tuple = TupMap_t<add_uniqueptr_t,
                                             output_tuple>; // the outputs as std::unique_ptrs
    using graph_ptr_tuple = TupMap_t<std::add_pointer_t,
                                     const_graph_tup>; // the graph as pointer

    using var_input_tuple = TupFilter_t<is_boxed_const, var_args>; // variadic input tuple
    using var_output_tuple = TupFilter_t<is_boxed_nonconst, var_args>; // variadic output tuple

    //using output_unboxed_tuple = TupFilter_t<is_unboxed_val,output_tuple>;				// the unboxed ones
    //using input_unboxed_tuple = TupFilter_t<is_unboxed_val,input_tuple>;				// the unboxed ones
    //using output_boxed_tuple = TupFilter_t<is_boxed_val,output_tuple>;				// the unboxed ones
    //using input_boxed_tuple = TupFilter_t<is_boxed_val,input_tuple>;				// the unboxed ones
    //using input_unboxed_ptr_tuple = TupMap_t<std::add_pointer_t,input_unboxed_tuple>;
    //using output_unboxed_ptr_tuple = TupMap_t<std::add_pointer_t,output_unboxed_tuple>;
    static constexpr size_t n_inputs = std::tuple_size<input_tuple>::value; // number of inputs
    static constexpr size_t n_outputs = std::tuple_size<output_tuple>::value; // number of outputs
    static constexpr bool has_graph = (std::tuple_size<const_graph_tup>::value > 0); // does it have a graph operand?

    //a string in the form of "@t1.t2.t3"... where t1,t2,t3,etc are the typenames of the input arguments as defined by DEFINE_TYPENAME
    static constexpr auto nameArray =
            GetTypeNames<all_args_tuple>(std::make_index_sequence<std::tuple_size_v<all_args_tuple>>{});
    static constexpr const char *inputTypeNames = nameArray.data();
};

template <auto F> struct ArgsTuples2 : public ArgsTuples<std::remove_pointer_t<decltype(F)>> {
};

// contains_type< tuple<a,b,c>, x >::value: true if x is in a,b,c ...
// no 'remove ref' etc is done.
template <typename TUPLET, typename T> struct contains_type {
};

template <typename T> struct contains_type<std::tuple<>, T> {
    static const bool value = false; // empty tuple contains nothing
};
/*
template <typename TA, typename T>
struct contains_type< std::tuple<TA>, T > {
	static const bool value = std::is_same<TA,T>::value;
};
*/
template <typename T, typename... TX> struct contains_type<std::tuple<T, TX...>, T> {
    static const bool value = true;
};
template <typename TA, typename... TX, typename T> struct contains_type<std::tuple<TA, TX...>, T> {
    static const bool value = contains_type<std::tuple<TX...>, T>::value;
};
template <typename TUPLET, typename T> struct not_contains_type {
    static const bool value = !contains_type<TUPLET, T>::value;
};

/*
 * Generic template
 */
template <typename... T> struct Concat_struct;

/*
 * Make the name nice 
 */
template <typename... T> using Concat = typename Concat_struct<T...>::type;

/*
 * Specialized that actually does the work:
 * Given two containers (Containter template C) "A" and "B", concatenate A and B
 */

/*
 * EJP: FIXME: this works for concatenating two things.
 * But it should be straightforward to concatenate N things by recursion
 * Additionally, concatinating a single thing should evaluate to itself.
 */

template <template <typename...> typename C, typename... As, typename... Bs> struct Concat_struct<C<As...>, C<Bs...>> {
    using type = C<As..., Bs...>;
};

/*
 * Generic template 
 */
template <typename... T> struct Product_helper_struct;

/*
 * Make the name nice 
 */
template <typename... T> using Product_helper = typename Product_helper_struct<T...>::type;

/*
 * Product helper specialization:
 * Container "C"
 * A single container of types
 */
template <template <typename...> typename C, typename... As> struct Product_helper_struct<C<As...>> {
    using type = C<As...>;
};

/*
 * Product helper specialization:
 * Container "C"
 * Product with empty set is empty set always
 */

template <template <typename...> typename C, typename... Prefixes, typename... Rest>
struct Product_helper_struct<C<Prefixes...>, C<>, Rest...> {
    using type = C<>;
};

// The two functions below do the bulk of the work

/*
 * Product helper specialization
 * First Arg: a container of prefixes,
 * Second Arg: a single (containered) element to append to each prefix
 * Args...: All the rest of the work
 * 
 * Create a container of new prefixes by concatinating each prefix with the new element
 * Then recurse using these new prefixes with the rest of the work
 * 
 * This handles a single element.  
 * The element is containerized so that it also handles a container with a single element,
 * or the last element in a list.
 */

template <template <typename...> typename C, typename... Prefixes, typename Elem, typename... Rest>
struct Product_helper_struct<C<Prefixes...>, C<Elem>, Rest...> {
    using new_prefixes = C<Concat<Prefixes, C<Elem>>...>;
    using type = Product_helper<new_prefixes, Rest...>;
};

/*
 * Product helper specialization
 * First Arg: a container of prefixes,
 * Second Arg: More than one containered elements 
 * Args...: All the rest of the work
 * 
 * Create a first list with the first element off the second argument, and
 * create the list recursing with just the single containerized element
 *  (This will use the specialization above)
 * Then create a second list by recursing with the rest of the elements of the second argument
 * Finally, Concatenate these two lists.
 * 
 * EJP: I think maybe I'm starting to get the hang of these template things.
 * 
 */

template <template <typename...> typename C, typename... Prefixes, typename FirstElem, typename... RestElem,
          typename... Rest>
struct Product_helper_struct<C<Prefixes...>, C<FirstElem, RestElem...>, Rest...> {
    using type = Concat<Product_helper<C<Prefixes...>, C<FirstElem>, Rest...>,
                        Product_helper<C<Prefixes...>, C<RestElem...>, Rest...>>;
};

template <typename... T> struct Product_struct;

template <template <typename...> typename C> struct Product_struct<C<>> {
    using type = C<>;
};

#if 0
template <template <typename...> typename C, typename... First,
          typename... Rest>
struct Product_struct<C<C<First...>, Rest...>> {
    using type = Product_helper<C<C<First>...>, Rest...>;
};
#else
template <template <typename...> typename C, typename... Rest> struct Product_struct<C<Rest...>> {
    using type = Product_helper<C<C<>>, Rest...>;
};
#endif

template <typename... T> using Product = typename Product_struct<T...>::type;

template <typename IterT> struct pair_to_iterators : std::pair<IterT, IterT> {
    pair_to_iterators(std::pair<IterT, IterT> const &&iter_pair_in) : std::pair<IterT, IterT>(std::move(iter_pair_in))
    {
    }
    pair_to_iterators(std::pair<IterT, IterT> const &iter_pair_in) : std::pair<IterT, IterT>(iter_pair_in) {}
    IterT begin() const { return this->first; }
    IterT end() const { return this->second; }
    IterT cbegin() const { return this->first; }
    IterT cend() const { return this->second; }
};

// insert a numeric object in a sorted vector,
// unless it's already there. Useful instead of set<T>
// if sizeof(T) and n are fairly small (since it uses O(n) inserts)
// Returns true if the value was inserted, false if there was a dup.
template <typename T, typename Allocator> bool insert_ordered_no_dups(std::vector<T, Allocator> &vec, T const &value)
{
    if (vec.empty() || vec.back() < value) {
        vec.emplace_back(value);
    } else {
        int hi = vec.size();
        int lo = 0;
        T const *p = &vec[0];
        while (lo < hi) {
            int const mid = (lo + hi) / 2u;
            if (value < p[mid]) {
                hi = mid;
            } else if (p[mid] < value) {
                lo = mid + 1;
            } else {
                return false; // is a dup.
            }
        }
        vec.insert(vec.begin() + lo, value);
    }
    return true;
}

// generic binary search.
// ======
// This is coded in a form which tends to work well on hexagon;
// the representation of the remaining sublist as ptr,offset
// reduces the critical-path calculation, and the update is simple
// enough that it usually requires no conditional calculations.
//
// =====
// Look for k in arr[0..n-1], which must be in order.
// Returns address of the first element which is >=k; if n==0 or
// 'extr' is a key extractor, which reads a K from T.
// if all are < k, it returns &arr[n].
//  uses comparisons K < K,  and also K==K if CHECK_EQ
//
//  CHECK_EQ: includes an '==k' check in each iteration. Generally
//    this will be faster if comparisons are cheap and it's common
//    to actually find the value in the list. Note, if the list
//    contains more than one of k, the result may be any of these if
//    CHECK_EQ is true; always the first if CHECK_EQ is false.
//
//  LIN_THR: when the sublist is <= this, use a linear search.
//  This is much faster per iteration than the binary search, when
//  the key is just an int and the records are not much bigger than an int.
//  For a table of ints or similar, probably should be about 6.
//  If you set this to >=3 or so, CHECK_EQ should always be false
//  since the equality check is unlikely to hit, and it makes the loop longer.

template <bool CHECK_EQ, int LIN_THR, typename EXTRACTOR, typename T, typename K>
inline T const *array_search_ordered(T const *arr, int n, K const &k, EXTRACTOR extr)
{
    T const *p = arr;
    static constexpr int LTHR = (LIN_THR < 3) ? 0 : 3;

    // invariant : p[0..n-1] have not been examined
    //  all before p[0] are <k, all at p[n] and beyond are >=k
    while (n > LTHR) {
        int const nx = n - 1;
        n >>= 1;
        T const *const px = &p[n];
        K const &pxv = extr(*px);
        if constexpr (CHECK_EQ)
            if (pxv == k) return px;
        if (pxv < k) {
            p = px + 1; // select part of list after px
            n = nx >> 1;
        }
    }
    if constexpr (LTHR > 0) {
        T const *const p_end = &p[n];
        while (p < p_end && extr(*p) < k)
            ++p;
    }
    return p;
}

// same with no EXTRACTOR
template <bool CHECK_EQ, int LIN_THR, typename T> T const *array_search_ordered(T const *arr, int n, T const &k)
{
    return array_search_ordered<CHECK_EQ, LIN_THR>(arr, n, k, [](T const &x) { return x; });
}

// same with T *
template <bool CHECK_EQ, int LIN_THR, typename EXTRACTOR, typename T, typename K>
T *array_search_ordered(T *arr, int n, K const &k, EXTRACTOR extr)
{
    return const_cast<T *>(
            array_search_ordered<CHECK_EQ, LIN_THR, EXTRACTOR, T>(const_cast<T const *>(arr), n, k, extr));
}
// with T *, no extractor
template <bool CHECK_EQ, int LIN_THR, typename T> T *array_search_ordered(T *arr, int n, T const &k)
{
    return const_cast<T *>(
            array_search_ordered<CHECK_EQ, LIN_THR>(const_cast<T const *>(arr), n, k, [](T const &x) { return x; }));
}

// given a std::vector<pair<T1,T2>> - which must be sorted in increasing
// order of T1 - look up a key (T1) value 'k', and return a pointer to the matching
// T2 - or null if not found.
// This makes use of T1 < T1 and T1==T1
template <typename T1, typename T2> T2 const *sorted_pair_lookup(std::vector<std::pair<T1, T2>> const &v, T1 const &k)
{
    int n = v.size();
    auto const *arrp = v.data();
    auto const *posn = array_search_ordered<false, 6>(arrp, n, k, [](decltype(*arrp) const &p) { return p.first; });
    if (posn >= &arrp[n] || posn->first != k) return nullptr;
    return &posn->second;
}

// lookup in a vector of sorted tuples; based on the first element.
// Returns pointer to the whole matching tuple, or nullptr.
template <typename T1, typename... Tx>
std::tuple<T1, Tx...> const *sorted_tuple_lookup(std::vector<std::tuple<T1, Tx...>> const &v, T1 const &k)
{
    int n = v.size();
    auto const *arrp = v.data();
    auto const *posn =
            array_search_ordered<false, 4>(arrp, n, k, [](decltype(*arrp) const &tup) { return std::get<0>(tup); });
    if (posn >= &arrp[n] || std::get<0>(*posn) != k) return nullptr;
    return posn;
}

} // namespace hnnx

#endif
