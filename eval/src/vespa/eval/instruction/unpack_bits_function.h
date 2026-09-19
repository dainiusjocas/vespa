// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/node_types.h>
#include <vespa/eval/eval/tensor_function.h>

namespace vespalib::eval {

/**
 * Result of matching a tensor-generating lambda against the
 * 'unpack_bits' expression shape (see UnpackBitsFunction below).
 **/
struct UnpackBitsMatch {
    const bool       is_unpack_bits;
    const bool       is_big_bitorder;
    const ValueType& src_type;
};

/**
 * Checks whether a tensor-generating lambda (as found inside a
 * Lambda or MapSubspaces tensor function node) has the shape of an
 * 'unpack_bits' expression, as documented on UnpackBitsFunction
 * below. Used both by UnpackBitsFunction::optimize and by other
 * optimizers wanting to recognize (and fuse) the same pattern
 * before it has been rewritten into an UnpackBitsFunction node.
 **/
UnpackBitsMatch detect_unpack_bits(const ValueType& dst_type, size_t num_bindings, const Function& lambda,
                                   const NodeTypes& types);

/**
 * Tensor function unpacking bits into separate values.
 *
 * The tensor containing the packed bits must be a vector (dense
 * tensor with 1 dimension) with cell type 'int8'. Bytes must be
 * processed with increasing index. Bits may be unpacked in either
 * 'big' or 'little' order. The result must be a vector (dense tensor
 * with 1 dimension) where the dimension is 8 times larger than the
 * input (since there are 8 bits packed into each int8 value).
 *
 * Baseline expression for 'big' bitorder (most significant bit first):
 * (Note: this is the default order used by numpy unpack_bits)
 * 'tensor<int8>(x[64])(bit(packed{x:(x/8)},7-(x%8)))'
 *
 * Baseline expression for 'little' bitorder (least significant bit first):
 * (Note: make sure this is the actual order of your bits)
 * 'tensor<int8>(x[64])(bit(packed{x:(x/8)},x%8))'
 **/
class UnpackBitsFunction : public tensor_function::Op1 {
private:
    bool _big_bitorder;

public:
    UnpackBitsFunction(const ValueType& res_type_in, const TensorFunction& packed, bool big);
    InterpretedFunction::Instruction compile_self(const ValueBuilderFactory& factory, Stash& stash) const override;
    bool result_is_mutable() const override { return true; }
    static const TensorFunction& optimize(const TensorFunction& expr, Stash& stash);
};

} // namespace vespalib::eval
