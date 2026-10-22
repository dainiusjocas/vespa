// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/eval/eval/llvm/compile_cache.h>
#include <vespa/searchlib/fef/blueprint.h>

namespace search::features {

/**
 * Blueprint for the elementwise element similarity ranking feature. It is created by the setup member function in the
 * elementwise ranking feature blueprint.
 *
 * This blueprint expects 3 parameters: index field name, dimension name and cell type.
 *
 * Example usage: elementwise(elementSimilarity(i),x,float) cause the elementwise ranking feature blueprint to create
 * this blueprint with parameters (i,x,float) and proxy calls to createExecutor() to this blueprint. The executor
 * returned by createExecutor() will create a tensor with a single mapped dimension 'x' that contains the element
 * similarity score for each element in the field 'i' matched by at least one query term searching the field.
 *
 * The per-element score is calculated by the same expression as used by the default output of the element similarity
 * feature (with parameters p, o, q, f and w), but without aggregating across elements. The expression is taken from
 * the rank property 'elementwise(elementSimilarity(i),x,float).output.default' if present, otherwise from
 * 'elementSimilarity(i).output.default', otherwise the built-in default is used. The value must be on the form
 * 'aggregator(expression)' (e.g. 'max((0.35*p+0.15*o+0.30*q+0.20*f)*w)'); the aggregator is validated but not used,
 * since there is exactly one score per element. Named outputs of the element similarity feature (e.g.
 * 'elementSimilarity(i).proximity') are not supported.
 */
class ElementwiseElementSimilarityBlueprint : public fef::Blueprint {
    const fef::FieldInfo*                   _field;
    vespalib::eval::CompileCache::Token::UP _compile_token;
    vespalib::eval::ValueType               _output_tensor_type;
    std::unique_ptr<vespalib::eval::Value>  _empty_output;

public:
    ElementwiseElementSimilarityBlueprint();
    ~ElementwiseElementSimilarityBlueprint() override;
    void visitDumpFeatures(const fef::IIndexEnvironment& env, fef::IDumpFeatureVisitor& visitor) const override;
    std::unique_ptr<fef::Blueprint> createInstance() const override;
    fef::ParameterDescriptions getDescriptions() const override;
    bool setup(const fef::IIndexEnvironment& env, const fef::ParameterList& params) override;
    fef::FeatureExecutor& createExecutor(const fef::IQueryEnvironment& env, vespalib::Stash& stash) const override;
};

} // namespace search::features
