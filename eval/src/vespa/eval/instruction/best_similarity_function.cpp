// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "best_similarity_function.h"

#include "unpack_bits_function.h"

#include <vespa/eval/eval/inline_operation.h>
#include <vespa/eval/eval/value.h>
#include <vespa/vespalib/util/binary_hamming_distance.h>

#include <optional>

namespace vespalib::eval {

using namespace tensor_function;
using namespace operation;
using vespalib::eval::nodes::TensorLambda;

namespace {

struct BestSimParam {
    ValueType res_type;
    size_t    inner_size;
    BestSimParam(const ValueType& res_type_in, size_t inner_size_in)
        : res_type(res_type_in), inner_size(inner_size_in) {}
};

struct UseDotProduct {
    static float calc(const float* pri, const float* sec, size_t size) {
        return DotProduct<float, float>::apply(pri, sec, size);
    }
};

struct UseHammingDist {
    static float calc(const Int8Float* pri, const Int8Float* sec, size_t size) {
        return binary_hamming_distance(pri, sec, size);
    }
};

template <typename CT, typename AGGR, typename DIST>
float best_similarity(const CT* pri, std::span<const CT> sec_cells, size_t inner_size) {
    AGGR aggr;
    for (const CT* sec = sec_cells.data(); sec < sec_cells.data() + sec_cells.size(); sec += inner_size) {
        aggr.sample(DIST::calc(pri, sec, inner_size));
    }
    return aggr.result();
}

template <bool is_double> const Value& create_empty_result(const ValueType& type, Stash& stash) {
    if (is_double) {
        return stash.create<DoubleValue>(0.0);
    } else if (type.count_mapped_dimensions() == 0) {
        auto zero_cells = stash.create_array<float>(type.dense_subspace_size());
        return stash.create<ValueView>(type, TrivialIndex::get(), TypedCells(zero_cells));
    } else {
        return stash.create<ValueView>(type, EmptyIndex::get(), TypedCells(nullptr, CellType::FLOAT, 0));
    }
}

template <bool is_double, typename CT, typename AGGR, typename DIST>
void my_best_similarity_op(InterpretedFunction::State& state, uint64_t param) {
    size_t           inner_size = is_double ? param : unwrap_param<BestSimParam>(param).inner_size;
    const ValueType& res_type = is_double ? DoubleValue::shared_type() : unwrap_param<BestSimParam>(param).res_type;
    const Value&     pri_value = state.peek(1);
    auto             pri_cells = pri_value.cells().typify<CT>();
    auto             sec_cells = state.peek(0).cells().typify<CT>();
    if ((pri_cells.size() == 0) || (sec_cells.size() == 0)) {
        return state.pop_pop_push(create_empty_result<is_double>(res_type, state.stash));
    }
    if (is_double) {
        auto best_sim = best_similarity<CT, AGGR, DIST>(pri_cells.data(), sec_cells, inner_size);
        return state.pop_pop_push(state.stash.create<DoubleValue>(best_sim));
    }
    auto      out_cells = state.stash.create_uninitialized_array<float>(pri_cells.size() / inner_size);
    const CT* pri = pri_cells.data();
    for (auto& out : out_cells) {
        out = best_similarity<CT, AGGR, DIST>(pri, sec_cells, inner_size);
        pri += inner_size;
    }
    Value& result_ref = state.stash.create<ValueView>(res_type, pri_value.index(), TypedCells(out_cells));
    state.pop_pop_push(result_ref);
}

// Asymmetric case: 'pri' is a full-precision (float/double) query vector, 'sec' is a
// bit-packed int8 tensor that would otherwise need 'unpack_bits' applied to it first. The
// packed bits are never materialized as unpacked cells; each per-subspace similarity is
// computed directly against the packed bytes via the SIMD-accelerated BitDotProduct. Unlike
// my_best_similarity_op, 'pri' and 'sec' step at different strides (elements vs. bytes) and
// have different cell types, so this cannot reuse best_similarity<CT,AGGR,DIST>().
template <bool is_double, typename QCT, bool big>
void my_best_similarity_unpack_bits_op(InterpretedFunction::State& state, uint64_t param) {
    size_t           inner_size = is_double ? param : unwrap_param<BestSimParam>(param).inner_size;
    const ValueType& res_type = is_double ? DoubleValue::shared_type() : unwrap_param<BestSimParam>(param).res_type;
    const Value&     pri_value = state.peek(1);
    auto             pri_cells = pri_value.cells().typify<QCT>();
    auto             sec_cells = state.peek(0).cells().typify<Int8Float>();
    if ((pri_cells.size() == 0) || (sec_cells.size() == 0)) {
        return state.pop_pop_push(create_empty_result<is_double>(res_type, state.stash));
    }
    size_t     byte_stride = inner_size / 8;
    const auto one_best = [&](const QCT* pri) noexcept {
        aggr::Max<double> aggr;
        for (const Int8Float* sec = sec_cells.data(); sec < sec_cells.data() + sec_cells.size(); sec += byte_stride) {
            aggr.sample(BitDotProduct<QCT>::apply(pri, sec, inner_size, big));
        }
        return aggr.result();
    };
    if (is_double) {
        return state.pop_pop_push(state.stash.create<DoubleValue>(one_best(pri_cells.data())));
    }
    auto       out_cells = state.stash.create_uninitialized_array<float>(pri_cells.size() / inner_size);
    const QCT* pri = pri_cells.data();
    for (auto& out : out_cells) {
        out = float(one_best(pri));
        pri += inner_size;
    }
    Value& result_ref = state.stash.create<ValueView>(res_type, pri_value.index(), TypedCells(out_cells));
    state.pop_pop_push(result_ref);
}

InterpretedFunction::op_function select_unpack_bits_op(bool is_double_result, CellType query_cell_type, bool big) {
    if (query_cell_type == CellType::FLOAT) {
        if (is_double_result) {
            return big ? my_best_similarity_unpack_bits_op<true, float, true>
                      : my_best_similarity_unpack_bits_op<true, float, false>;
        }
        return big ? my_best_similarity_unpack_bits_op<false, float, true>
                  : my_best_similarity_unpack_bits_op<false, float, false>;
    }
    if (is_double_result) {
        return big ? my_best_similarity_unpack_bits_op<true, double, true>
                  : my_best_similarity_unpack_bits_op<true, double, false>;
    }
    return big ? my_best_similarity_unpack_bits_op<false, double, true>
              : my_best_similarity_unpack_bits_op<false, double, false>;
}

bool is_plain_dense_operand(const ValueType& type) {
    return type.is_dense() && (type.count_indexed_dimensions() == 1) &&
           ((type.cell_type() == CellType::FLOAT) || (type.cell_type() == CellType::DOUBLE));
}

struct UnpackBitsOperand {
    const TensorFunction& packed;
    ValueType             conceptual_type; // still has the mapped/indexed 'best' dimension
    bool                  big_bitorder;
};

// Recognizes the same raw (not-yet-rewritten) 'unpack_bits' shapes as UnpackBitsFunction::optimize
// and DenseUnpackBitsDotProductFunction::optimize: a plain dense Lambda, or a MapSubspaces wrapping
// a per-subspace TensorLambda. Must run before UnpackBitsFunction/DenseUnpackBitsDotProductFunction
// (see optimize_tensor_function.cpp) to still see these raw shapes.
std::optional<UnpackBitsOperand> match_unpack_bits_operand(const TensorFunction& expr, Stash& stash) {
    if (auto lambda = as<Lambda>(expr)) {
        auto detected =
            detect_unpack_bits(lambda->result_type(), lambda->bindings().size(), lambda->lambda(), lambda->types());
        if (detected.is_unpack_bits) {
            assert(lambda->bindings().size() == 1);
            const TensorFunction& packed = inject(detected.src_type, lambda->bindings()[0], stash);
            return UnpackBitsOperand{packed, lambda->result_type(), detected.is_big_bitorder};
        }
    } else if (auto map_subspaces = as<MapSubspaces>(expr)) {
        if (auto tensor_lambda = as<TensorLambda>(map_subspaces->lambda().root())) {
            auto detected = detect_unpack_bits(tensor_lambda->type(), tensor_lambda->bindings().size(),
                                               tensor_lambda->lambda(), map_subspaces->types());
            if (detected.is_unpack_bits) {
                return UnpackBitsOperand{map_subspaces->child(), map_subspaces->result_type(),
                                         detected.is_big_bitorder};
            }
        }
    }
    return std::nullopt;
}

//-----------------------------------------------------------------------------

size_t stride(const ValueType& type, const std::string& name) {
    size_t stride = 0;
    for (const auto& dim : type.dimensions()) {
        if (dim.is_indexed()) {
            if (dim.name == name) {
                stride = 1;
            } else {
                stride *= dim.size;
            }
        }
    }
    return stride;
}

bool check_dims(const ValueType& pri, const ValueType& sec, const std::string& best, const std::string& inner) {
    if ((stride(pri, inner) != 1) || (stride(sec, inner) != 1)) {
        return false;
    }
    if (pri.dimension_index(best) != ValueType::Dimension::npos) {
        return false;
    }
    if (sec.dimension_index(best) == ValueType::Dimension::npos) {
        return false;
    }
    for (auto&& type = sec.reduce({inner, best}); auto&& dim : type.dimensions()) {
        if (!dim.is_trivial()) {
            return false;
        }
    }
    return true;
}

size_t get_dim_size(const ValueType& type, const std::string& dim) {
    size_t npos = ValueType::Dimension::npos;
    size_t idx = type.dimension_index(dim);
    assert(idx != npos);
    assert(type.dimensions()[idx].is_indexed());
    return type.dimensions()[idx].size;
}

// Tries to fuse the asymmetric case: 'plain' is a plain dense float/double query vector,
// 'packed_candidate' is (or should be) shaped like 'unpack_bits(<int8 tensor>, ...)'. On
// success, the resulting BestSimilarityFunction is built with the *packed* int8 tensor
// function as its 'sec' child directly, so the conceptual unpacked tensor is never created.
const TensorFunction* try_unpack_bits_maxsim(const ValueType& res_type, const TensorFunction& plain,
                                             const TensorFunction& packed_candidate, const std::string& best_dim,
                                             const std::string& inner_dim, Stash& stash) {
    if (!is_plain_dense_operand(plain.result_type())) {
        return nullptr;
    }
    auto match = match_unpack_bits_operand(packed_candidate, stash);
    if (!match || !check_dims(plain.result_type(), match->conceptual_type, best_dim, inner_dim)) {
        return nullptr;
    }
    size_t inner_size = get_dim_size(match->conceptual_type, inner_dim);
    auto op = select_unpack_bits_op(res_type.is_double(), plain.result_type().cell_type(), match->big_bitorder);
    return &stash.create<BestSimilarityFunction>(res_type, plain, match->packed, op, inner_size);
}

const Reduce* check_reduce(const TensorFunction& expr, std::initializer_list<Aggr> allow) {
    if (auto reduce = as<Reduce>(expr)) {
        if (reduce->dimensions().size() == 1) {
            if (std::find(allow.begin(), allow.end(), reduce->aggr()) != allow.end()) {
                return reduce;
            }
        }
    }
    return nullptr;
}

const Join* check_join(const TensorFunction& expr, std::initializer_list<op2_t> allow) {
    if (auto join = as<Join>(expr)) {
        if (std::find(allow.begin(), allow.end(), join->function()) != allow.end()) {
            return join;
        }
    }
    return nullptr;
}

struct SelectFun {
    const ValueType& res_type;
    const ValueType& lhs_type;
    const ValueType& rhs_type;
    template <typename ResType, typename LhsType, typename RhsType>
    SelectFun(const ResType& res, const LhsType& lhs, const RhsType& rhs)
        : res_type(res.result_type()), lhs_type(lhs.result_type()), rhs_type(rhs.result_type()) {}
    template <typename R1>
    static InterpretedFunction::op_function invoke(Aggr best_aggr, op2_t join_fun, CellType cell_types) {
        if ((best_aggr == Aggr::MAX) && (join_fun == Mul::f) && (cell_types == CellType::FLOAT)) {
            return my_best_similarity_op<R1::value, float, aggr::Max<float>, UseDotProduct>;
        }
        if ((best_aggr == Aggr::MIN) && (join_fun == Hamming::f) && (cell_types == CellType::INT8)) {
            return my_best_similarity_op<R1::value, Int8Float, aggr::Min<float>, UseHammingDist>;
        }
        return nullptr;
    }
    InterpretedFunction::op_function operator()(Aggr best_aggr, op2_t join_fun) {
        static_assert(std::is_same_v<float, CellValueType<CellType::FLOAT>>);
        static_assert(std::is_same_v<Int8Float, CellValueType<CellType::INT8>>);
        if (lhs_type.cell_type() != rhs_type.cell_type()) {
            return nullptr;
        }
        return typify_invoke<1, TypifyBool, SelectFun>(res_type.is_double(), best_aggr, join_fun,
                                                       lhs_type.cell_type());
    }
};

} // namespace

uint64_t BestSimilarityFunction::make_param(Stash& stash) const {
    if (result_type().is_double()) {
        return _inner_size;
    }
    return wrap_param<BestSimParam>(stash.create<BestSimParam>(result_type(), _inner_size));
}

BestSimilarityFunction::BestSimilarityFunction(const ValueType& res_type_in, const TensorFunction& pri,
                                               const TensorFunction& sec, InterpretedFunction::op_function my_fun,
                                               size_t inner_size)
    : tensor_function::Op2(res_type_in, pri, sec), _my_fun(my_fun), _inner_size(inner_size) {
}

InterpretedFunction::Instruction BestSimilarityFunction::compile_self(const ValueBuilderFactory&,
                                                                      Stash& stash) const {
    return InterpretedFunction::Instruction(_my_fun, make_param(stash));
}

const TensorFunction& BestSimilarityFunction::optimize(const TensorFunction& expr, Stash& stash) {
    if (auto best_reduce = check_reduce(expr, {Aggr::MAX, Aggr::MIN})) {
        if (auto sum_reduce = check_reduce(best_reduce->child(), {Aggr::SUM})) {
            if (auto join = check_join(sum_reduce->child(), {Mul::f, Hamming::f})) {
                const auto&           best_dim = best_reduce->dimensions()[0];
                const auto&           inner_dim = sum_reduce->dimensions()[0];
                const TensorFunction& lhs = join->lhs();
                const TensorFunction& rhs = join->rhs();
                SelectFun select_fun(expr, lhs, rhs);
                if (auto my_fun = select_fun(best_reduce->aggr(), join->function())) {
                    if (check_dims(lhs.result_type(), rhs.result_type(), best_dim, inner_dim)) {
                        size_t inner_size = get_dim_size(lhs.result_type(), inner_dim);
                        return stash.create<BestSimilarityFunction>(expr.result_type(), lhs, rhs, my_fun, inner_size);
                    }
                    if (check_dims(rhs.result_type(), lhs.result_type(), best_dim, inner_dim)) {
                        size_t inner_size = get_dim_size(rhs.result_type(), inner_dim);
                        return stash.create<BestSimilarityFunction>(expr.result_type(), rhs, lhs, my_fun, inner_size);
                    }
                } else if ((best_reduce->aggr() == Aggr::MAX) && (join->function() == Mul::f)) {
                    // Neither operand matched a same-cell-type similarity measure; check for
                    // the asymmetric float-query/int8-packed-document (unpack_bits) case.
                    if (auto* result =
                            try_unpack_bits_maxsim(expr.result_type(), lhs, rhs, best_dim, inner_dim, stash)) {
                        return *result;
                    }
                    if (auto* result =
                            try_unpack_bits_maxsim(expr.result_type(), rhs, lhs, best_dim, inner_dim, stash)) {
                        return *result;
                    }
                }
            }
        }
    }
    return expr;
}

} // namespace vespalib::eval
