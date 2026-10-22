// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "element_similarity_calculator.h"
#include "elementwise_output.h"

#include <vespa/searchlib/fef/featureexecutor.h>
#include <vespa/vespalib/stllike/hash_map.h>

namespace search::fef {
class IQueryEnvironment;
}

namespace search::features {

/**
 * Executor for the elementwise element similarity feature over a single index field. It calculates the element
 * similarity score for each element in the field matched by at least one query term searching the field. These scores
 * are then used to build an output tensor with a single mapped dimension, with element id as label value and element
 * similarity score as tensor cell value.
 */
class ElementwiseElementSimilarityExecutor : public fef::FeatureExecutor {
public:
    using ScoreFunction = double (*)(double, double, double, double, double);

private:
    ElementSimilarityCalculator          _calculator;
    ScoreFunction                        _score_function;
    vespalib::hash_map<uint32_t, double> _scores; // element id -> score
    ElementwiseOutput                    _output;

public:
    ElementwiseElementSimilarityExecutor(const fef::IQueryEnvironment& env, uint32_t field_id,
                                         ScoreFunction score_function, const vespalib::eval::Value& empty_output);
    ~ElementwiseElementSimilarityExecutor() override;
    bool isPure() override { return !_calculator.has_terms(); }
    void handle_bind_match_data(const fef::MatchData& match_data) override;
    void execute(uint32_t docId) override;
};

} // namespace search::features
