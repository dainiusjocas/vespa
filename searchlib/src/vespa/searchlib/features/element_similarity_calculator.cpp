// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "element_similarity_calculator.h"

#include <vespa/searchlib/fef/iqueryenvironment.h>
#include <vespa/searchlib/fef/itermdata.h>
#include <vespa/searchlib/fef/matchdata.h>

namespace search::features {

const std::vector<std::string>& ElementSimilarityCalculator::expression_params() {
    static const std::vector<std::string> params({"p", "o", "q", "f", "w"});
    return params;
}

const std::string& ElementSimilarityCalculator::default_output_expression() {
    static const std::string expression("max((0.35*p+0.15*o+0.30*q+0.20*f)*w)");
    return expression;
}

ElementSimilarityCalculator::QueryTerms ElementSimilarityCalculator::collect_query_terms(
    const fef::IQueryEnvironment& env, uint32_t field_id) {
    QueryTerms terms;
    for (uint32_t i = 0; i < env.getNumTerms(); ++i) {
        const fef::ITermData* termData = env.getTerm(i);
        if (termData->getWeight().percent() != 0) { // only consider query terms with contribution
            using FRA = fef::ITermFieldRangeAdapter;
            for (FRA iter(*termData); iter.valid(); iter.next()) {
                const fef::ITermFieldData& tfd = iter.get();
                if (tfd.getFieldId() == field_id) {
                    int term_weight = termData->getWeight().percent();
                    terms.total_weight += term_weight;
                    terms.handles.push_back(tfd.getHandle());
                    terms.weights.push_back(term_weight);
                }
            }
        }
    }
    return terms;
}

ElementSimilarityCalculator::ElementSimilarityCalculator(const fef::IQueryEnvironment& env, uint32_t field_id)
    : _terms(collect_query_terms(env, field_id)),
      _pos(_terms.handles.size(), nullptr),
      _end(_terms.handles.size(), nullptr),
      _position_queue(CmpPosition(_pos.data())),
      _element_queue(CmpElement(_pos.data())),
      _md(nullptr) {
}

ElementSimilarityCalculator::~ElementSimilarityCalculator() = default;

void ElementSimilarityCalculator::prepare(uint32_t docid) {
    for (size_t i = 0; i < _terms.handles.size(); ++i) {
        const fef::TermFieldMatchData* tfmd = _md->resolveTermField(_terms.handles[i]);
        if (tfmd->has_ranking_data(docid)) {
            _pos[i] = tfmd->begin();
            _end[i] = tfmd->end();
            if (_pos[i] != _end[i]) {
                _element_queue.push(i);
            }
        }
    }
}

} // namespace search::features
