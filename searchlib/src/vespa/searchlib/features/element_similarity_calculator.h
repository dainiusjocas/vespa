// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "element_similarity_state.h"

#include <vespa/searchlib/fef/handle.h>
#include <vespa/searchlib/fef/termfieldmatchdata.h>
#include <vespa/vespalib/util/priority_queue.h>

#include <string>
#include <vector>

namespace search::fef {
class IQueryEnvironment;
class MatchData;
} // namespace search::fef

namespace search::features {

/*
 * Calculates the per-element similarity state (proximity, order, query coverage, field coverage and element weight)
 * for each element in a single field matched by the query terms searching that field. Shared by the element
 * similarity feature (aggregating across elements) and the elementwise element similarity feature (one tensor cell
 * per element).
 */
class ElementSimilarityCalculator {
private:
    using ITR = fef::TermFieldMatchData::PositionsIterator;

    struct CmpPosition {
        ITR* pos;

        CmpPosition(ITR* pos_in) : pos(pos_in) {}

        bool operator()(uint16_t a, uint16_t b) {
            return (pos[a]->getPosition() == pos[b]->getPosition()) ? (a < b)
                                                                    : (pos[a]->getPosition() < pos[b]->getPosition());
        }
    };

    struct CmpElement {
        ITR* pos;

        CmpElement(ITR* pos_in) : pos(pos_in) {}

        bool operator()(uint16_t a, uint16_t b) { return pos[a]->getElementId() < pos[b]->getElementId(); }
    };

    using PositionQueue = vespalib::PriorityQueue<uint16_t, CmpPosition>;
    using ElementQueue = vespalib::PriorityQueue<uint16_t, CmpElement>;

    struct QueryTerms {
        std::vector<fef::TermFieldHandle> handles;
        std::vector<int>                  weights;
        int                               total_weight = 0;
    };
    static QueryTerms collect_query_terms(const fef::IQueryEnvironment& env, uint32_t field_id);

    QueryTerms _terms;
    // The queue comparators point into _pos, so _pos must never be resized after construction
    std::vector<ITR>      _pos;
    std::vector<ITR>      _end;
    PositionQueue         _position_queue;
    ElementQueue          _element_queue;
    const fef::MatchData* _md;

    // take the front term in the position queue,
    // iterate to its next element (or end), and
    // put it back into the element queue if possible
    void requeue_pos_front(uint32_t element) {
        uint16_t term = _position_queue.front();
        _position_queue.pop_front();
        while (_pos[term] != _end[term] && (_pos[term]->getElementId() == element)) {
            ++_pos[term];
        }
        if (_pos[term] != _end[term]) {
            _element_queue.push(term);
        }
    }

    void prepare(uint32_t docid);

public:
    // The parameter names available in per-element expressions
    static const std::vector<std::string>& expression_params();
    // The default output expression, aggregating per-element scores across elements
    static const std::string& default_output_expression();

    ElementSimilarityCalculator(const fef::IQueryEnvironment& env, uint32_t field_id);
    ElementSimilarityCalculator(const ElementSimilarityCalculator&) = delete;
    ElementSimilarityCalculator& operator=(const ElementSimilarityCalculator&) = delete;
    ~ElementSimilarityCalculator();

    bool has_terms() const noexcept { return !_terms.handles.empty(); }
    void bind_match_data(const fef::MatchData& md) noexcept { _md = &md; }

    /*
     * Calls on_element(element_id, state) for each element (in increasing element id order) matched by at least one
     * query term searching the field. The state has its scores calculated.
     */
    template <typename OnElement> void calculate(uint32_t docid, OnElement&& on_element) {
        prepare(docid);
        while (!_element_queue.empty()) {
            uint32_t element_id = _pos[_element_queue.front()]->getElementId();
            while (!_element_queue.empty() && _pos[_element_queue.front()]->getElementId() == element_id) {
                _position_queue.push(_element_queue.front());
                _element_queue.pop_front();
            }
            uint16_t               first = _position_queue.front();
            ElementSimilarityState state(_pos[first]->getElementLen(), _pos[first]->getElementWeight(),
                                         _pos[first]->getPosition(), _terms.weights[first], first);
            requeue_pos_front(element_id);
            while (!_position_queue.empty()) {
                uint16_t item = _position_queue.front();
                if (state.want_match(_pos[item]->getPosition())) {
                    state.addMatch(_pos[item]->getPosition(), _terms.weights[item], item);
                    requeue_pos_front(element_id);
                } else {
                    ++_pos[item];
                    if (_pos[item] == _end[item] || _pos[item]->getElementId() != element_id) {
                        requeue_pos_front(element_id);
                    } else {
                        _position_queue.adjust();
                    }
                }
            }
            state.calculate_scores(_terms.total_weight);
            on_element(element_id, state);
        }
    }
};

} // namespace search::features
