// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "elementwise_element_similarity_executor.h"

namespace search::features {

using vespalib::eval::Value;

ElementwiseElementSimilarityExecutor::ElementwiseElementSimilarityExecutor(const fef::IQueryEnvironment& env,
                                                                           uint32_t                      field_id,
                                                                           ScoreFunction score_function,
                                                                           const Value&  empty_output)
    : FeatureExecutor(),
      _calculator(env, field_id),
      _score_function(score_function),
      _scores(),
      _output(empty_output) {
}

ElementwiseElementSimilarityExecutor::~ElementwiseElementSimilarityExecutor() = default;

void ElementwiseElementSimilarityExecutor::handle_bind_match_data(const fef::MatchData& match_data) {
    _calculator.bind_match_data(match_data);
}

void ElementwiseElementSimilarityExecutor::execute(uint32_t doc_id) {
    _scores.clear();
    _calculator.calculate(doc_id, [this](uint32_t element_id, const ElementSimilarityState& state) {
        _scores.insert(std::make_pair(element_id, _score_function(state.proximity, state.order, state.query_coverage,
                                                                  state.field_coverage, state.element_weight)));
    });
    outputs().set_object(0, _output.build(_scores));
}

} // namespace search::features
