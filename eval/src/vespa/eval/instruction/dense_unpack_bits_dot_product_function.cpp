// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "dense_unpack_bits_dot_product_function.h"

#include "unpack_bits_function.h"

#include <vespa/eval/eval/inline_operation.h>
#include <vespa/eval/eval/value.h>

#include <cassert>

namespace vespalib::eval {

using namespace tensor_function;
using namespace operation;

namespace {

template <typename LCT, bool big> void my_bit_dot_product_op(InterpretedFunction::State& state, uint64_t) {
    auto   lhs_cells = state.peek(1).cells().typify<LCT>();
    auto   packed_cells = state.peek(0).cells().typify<Int8Float>();
    double result = BitDotProduct<LCT>::apply(lhs_cells.data(), packed_cells.data(), lhs_cells.size(), big);
    state.pop_pop_push(state.stash.create<DoubleValue>(result));
}

bool is_plain_dense_operand(const ValueType& type) {
    return type.is_dense() && (type.count_indexed_dimensions() == 1) &&
           ((type.cell_type() == CellType::FLOAT) || (type.cell_type() == CellType::DOUBLE));
}

} // namespace

DenseUnpackBitsDotProductFunction::DenseUnpackBitsDotProductFunction(const TensorFunction& lhs_in,
                                                                      const TensorFunction& packed_in,
                                                                      bool                  big_bitorder)
    : tensor_function::Op2(ValueType::double_type(), lhs_in, packed_in), _big_bitorder(big_bitorder) {
}

InterpretedFunction::Instruction DenseUnpackBitsDotProductFunction::compile_self(const ValueBuilderFactory&,
                                                                                 Stash&) const {
    auto lct = lhs().result_type().cell_type();
    assert((lct == CellType::FLOAT) || (lct == CellType::DOUBLE));
    auto op = (lct == CellType::FLOAT)
                  ? (_big_bitorder ? my_bit_dot_product_op<float, true> : my_bit_dot_product_op<float, false>)
                  : (_big_bitorder ? my_bit_dot_product_op<double, true> : my_bit_dot_product_op<double, false>);
    return InterpretedFunction::Instruction(op);
}

const TensorFunction& DenseUnpackBitsDotProductFunction::optimize(const TensorFunction& expr, Stash& stash) {
    auto reduce = as<Reduce>(expr);
    if (!reduce || (reduce->aggr() != Aggr::SUM) || !expr.result_type().is_double()) {
        return expr;
    }
    auto join = as<Join>(reduce->child());
    if (!join || (join->function() != Mul::f)) {
        return expr;
    }
    auto lhs_lambda = as<Lambda>(join->lhs());
    auto rhs_lambda = as<Lambda>(join->rhs());
    const Lambda*         packed_lambda;
    const TensorFunction* plain;
    if (lhs_lambda && !rhs_lambda) {
        packed_lambda = lhs_lambda;
        plain = &join->rhs();
    } else if (rhs_lambda && !lhs_lambda) {
        packed_lambda = rhs_lambda;
        plain = &join->lhs();
    } else {
        // Neither side (or, pathologically, both sides) look like an
        // unpack_bits expression; let DenseDotProductFunction (and later
        // UnpackBitsFunction) handle this the way they did before this
        // optimizer existed.
        return expr;
    }
    auto detected = detect_unpack_bits(packed_lambda->result_type(), packed_lambda->bindings().size(),
                                       packed_lambda->lambda(), packed_lambda->types());
    if (!detected.is_unpack_bits) {
        return expr;
    }
    const ValueType& unpacked_type = packed_lambda->result_type();
    if (!is_plain_dense_operand(plain->result_type()) ||
        (plain->result_type().dimensions() != unpacked_type.dimensions()))
    {
        return expr;
    }
    assert(packed_lambda->bindings().size() == 1);
    const TensorFunction& packed = inject(detected.src_type, packed_lambda->bindings()[0], stash);
    return stash.create<DenseUnpackBitsDotProductFunction>(*plain, packed, detected.is_big_bitorder);
}

} // namespace vespalib::eval
