// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "elementwise_element_similarity_blueprint.h"

#include "element_similarity_calculator.h"
#include "elementwise_element_similarity_executor.h"
#include "elementwise_utils.h"

#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/llvm/compiled_function.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/searchlib/fef/fieldinfo.h>
#include <vespa/searchlib/fef/properties.h>
#include <vespa/vespalib/util/stash.h>

namespace search::features {

using fef::FeatureType;
using vespalib::eval::CompileCache;
using vespalib::eval::FastValueBuilderFactory;
using vespalib::eval::Function;
using vespalib::eval::PassParams;
using vespalib::eval::ValueType;

namespace {

std::string element_similarity_feature_base_name("elementSimilarity");

std::string lookup_output_expression(const fef::Properties& props, const fef::ParameterList& params) {
    std::string suffix(".output.default");
    auto        elementwise_property =
        props.lookup(ElementwiseUtils::feature_name(element_similarity_feature_base_name, params) + suffix);
    if (elementwise_property.found()) {
        return elementwise_property.get();
    }
    return props.lookup(ElementwiseUtils::nested_feature_name(element_similarity_feature_base_name, params) + suffix)
        .get(ElementSimilarityCalculator::default_output_expression());
}

bool is_known_aggregator(const std::string& name) {
    return (name == "max") || (name == "avg") || (name == "sum");
}

} // namespace

ElementwiseElementSimilarityBlueprint::ElementwiseElementSimilarityBlueprint()
    : fef::Blueprint("elementwiseElementSimilarity"),
      _field(nullptr),
      _compile_token(),
      _output_tensor_type(ValueType::error_type()),
      _empty_output() {
}

ElementwiseElementSimilarityBlueprint::~ElementwiseElementSimilarityBlueprint() = default;

void ElementwiseElementSimilarityBlueprint::visitDumpFeatures(const fef::IIndexEnvironment&,
                                                              fef::IDumpFeatureVisitor&) const {
}

std::unique_ptr<fef::Blueprint> ElementwiseElementSimilarityBlueprint::createInstance() const {
    return std::make_unique<ElementwiseElementSimilarityBlueprint>();
}

fef::ParameterDescriptions ElementwiseElementSimilarityBlueprint::getDescriptions() const {
    return fef::ParameterDescriptions().desc().indexField(fef::ParameterCollection::ANY).string().string();
}

bool ElementwiseElementSimilarityBlueprint::setup(const fef::IIndexEnvironment& env,
                                                  const fef::ParameterList&     params) {
    _field = params[0].asField();
    auto        config = lookup_output_expression(env.getProperties(), params);
    std::string aggr_name;
    std::string expr;
    std::string error;
    if (!Function::unwrap(config, aggr_name, expr, error)) {
        return fail("could not extract aggregator and expression from config value '%s' (%s)", config.c_str(),
                    error.c_str());
    }
    if (!is_known_aggregator(aggr_name)) {
        return fail("unknown aggregator '%s' in config value '%s'", aggr_name.c_str(), config.c_str());
    }
    auto function = Function::parse(ElementSimilarityCalculator::expression_params(), expr);
    if (function->has_error()) {
        return fail("per-element expression parse error: %s", function->get_error().c_str());
    }
    _compile_token = CompileCache::compile(*function, PassParams::SEPARATE);
    auto fail_message =
        ElementwiseUtils::build_output_tensor_type(_output_tensor_type, params[1].getValue(), params[2].getValue());
    if (fail_message.has_value()) {
        return fail("%s", fail_message.value().c_str());
    }
    _empty_output = vespalib::eval::value_from_spec(_output_tensor_type.to_spec(), FastValueBuilderFactory::get());
    FeatureType output_type = FeatureType::object(_output_tensor_type);
    describeOutput("score",
                   "The element similarity score for each element in the given index field matched by the query",
                   output_type);
    return true;
}

fef::FeatureExecutor& ElementwiseElementSimilarityBlueprint::createExecutor(const fef::IQueryEnvironment& env,
                                                                            vespalib::Stash& stash) const {
    return stash.create<ElementwiseElementSimilarityExecutor>(env, _field->id(),
                                                              _compile_token->get().get_function<5>(), *_empty_output);
}

} // namespace search::features
