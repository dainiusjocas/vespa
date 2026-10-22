// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "element_similarity_feature.h"

#include "element_similarity_calculator.h"

#include <vespa/eval/eval/llvm/compile_cache.h>
#include <vespa/eval/eval/llvm/compiled_function.h>
#include <vespa/searchlib/fef/featurenamebuilder.h>
#include <vespa/searchlib/fef/itermdata.h>
#include <vespa/searchlib/fef/properties.h>
#include <vespa/vespalib/util/stash.h>

#include <vespa/log/log.h>
LOG_SETUP(".features.elementsimilarity");

namespace search::features {

using CollectionType = fef::FieldInfo::CollectionType;

namespace {

//-----------------------------------------------------------------------------

struct Aggregator {
    using UP = std::unique_ptr<Aggregator>;
    virtual UP create() const = 0;
    virtual void clear() = 0;
    virtual void add(double) = 0;
    virtual double get() const = 0;
    virtual ~Aggregator() = default;
};

struct MaxAggregator : Aggregator {
    size_t count;
    double value;

    MaxAggregator() : count(0), value(0.0) {}
    UP create() const override { return UP(new MaxAggregator()); }
    void clear() override {
        count = 0;
        value = 0.0;
    }
    void add(double v) override { value = ((++count == 1) || (v > value)) ? v : value; }
    double get() const override { return value; }
};

struct AvgAggregator : Aggregator {
    size_t count;
    double value;

    AvgAggregator() : count(0), value(0.0) {}

    UP create() const override { return UP(new AvgAggregator()); }
    void clear() override {
        count = 0;
        value = 0.0;
    }
    void add(double v) override {
        ++count;
        value += v;
    }
    double get() const override { return (count == 0) ? 0.0 : (value / count); }
};

struct SumAggregator : Aggregator {
    double value;

    SumAggregator() : value(0.0) {}
    UP create() const override { return UP(new SumAggregator()); }
    void clear() override { value = 0.0; }
    void add(double v) override { value += v; }
    double get() const override { return value; }
};

Aggregator::UP create_aggregator(const std::string& name) {
    if (name == "max") {
        return Aggregator::UP(new MaxAggregator());
    }
    if (name == "avg") {
        return Aggregator::UP(new AvgAggregator());
    }
    if (name == "sum") {
        return Aggregator::UP(new SumAggregator());
    }
    return Aggregator::UP(nullptr);
}

//-----------------------------------------------------------------------------

typedef double (*function_5)(double, double, double, double, double);

using OutputSpec = std::pair<function_5, Aggregator::UP>;

//-----------------------------------------------------------------------------

class ElementSimilarityExecutor : public fef::FeatureExecutor {
private:
    ElementSimilarityCalculator _calculator;
    std::vector<OutputSpec>     _outputs;

public:
    ElementSimilarityExecutor(const fef::IQueryEnvironment& env, uint32_t field_id,
                              std::vector<OutputSpec>&& outputs_in);
    ~ElementSimilarityExecutor() override;

    bool isPure() override { return !_calculator.has_terms(); }

    void handle_bind_match_data(const fef::MatchData& md) override { _calculator.bind_match_data(md); }

    void execute(uint32_t docId) override {
        for (auto& output : _outputs) {
            output.second->clear();
        }
        _calculator.calculate(docId, [this](uint32_t, const ElementSimilarityState& state) {
            for (auto& output : _outputs) {
                output.second->add(output.first(state.proximity, state.order, state.query_coverage,
                                                state.field_coverage, state.element_weight));
            }
        });
        for (size_t i = 0; i < _outputs.size(); ++i) {
            outputs().set_number(i, _outputs[i].second->get());
        }
    }
};

ElementSimilarityExecutor::ElementSimilarityExecutor(const fef::IQueryEnvironment& env, uint32_t field_id,
                                                     std::vector<OutputSpec>&&     outputs_in)
    : _calculator(env, field_id), _outputs(std::move(outputs_in)) {
}
ElementSimilarityExecutor::~ElementSimilarityExecutor() = default;

//-----------------------------------------------------------------------------

std::vector<std::pair<std::string, std::string>> extract_properties(const fef::Properties& props,
                                                                    const std::string&     ns,
                                                                    const std::string&     first_name,
                                                                    const std::string&     first_default) {
    struct MyVisitor : fef::IPropertiesVisitor {
        const std::string&                                first_name;
        std::vector<std::pair<std::string, std::string>>& result;

        MyVisitor(const std::string& first_name_in, std::vector<std::pair<std::string, std::string>>& result_in)
            : first_name(first_name_in), result(result_in) {}

        void visitProperty(const fef::Property::Value& key, const fef::Property& values) override {
            if (key != first_name) {
                result.emplace_back(key, values.get());
            }
        }
    };
    std::vector<std::pair<std::string, std::string>> result;
    result.emplace_back(first_name, props.lookup(ns, first_name).get(first_default));
    MyVisitor my_visitor(first_name, result);
    props.visitNamespace(ns, my_visitor);
    return result;
}

std::vector<std::pair<std::string, std::string>> get_outputs(const fef::Properties& props,
                                                             const std::string&     feature) {
    return extract_properties(props, feature + ".output", "default",
                              ElementSimilarityCalculator::default_output_expression());
}

} // namespace

//-----------------------------------------------------------------------------

struct ElementSimilarityBlueprint::OutputContext {
    vespalib::eval::CompileCache::Token::UP compile_token;
    Aggregator::UP                          aggregator_factory;

    OutputContext(const vespalib::eval::Function& function, Aggregator::UP aggregator)
        : compile_token(vespalib::eval::CompileCache::compile(function, vespalib::eval::PassParams::SEPARATE)),
          aggregator_factory(std::move(aggregator)) {}
};

//-----------------------------------------------------------------------------

ElementSimilarityBlueprint::ElementSimilarityBlueprint()
    : Blueprint("elementSimilarity"), _field_id(fef::IllegalFieldId), _outputs() {
}

ElementSimilarityBlueprint::~ElementSimilarityBlueprint() = default;

void ElementSimilarityBlueprint::visitDumpFeatures(const fef::IIndexEnvironment& env,
                                                   fef::IDumpFeatureVisitor&     visitor) const {
    for (uint32_t i = 0; i < env.getNumFields(); ++i) {
        const fef::FieldInfo& field = *env.getField(i);
        if ((field.type() == fef::FieldType::INDEX) && (field.collection() != CollectionType::SINGLE) &&
            (!field.isFilter()))
        {
            fef::FeatureNameBuilder fnb;
            fnb.baseName(getBaseName()).parameter(field.name());
            auto outputs = get_outputs(env.getProperties(), fnb.buildName());
            visitor.visitDumpFeature(fnb.output("").buildName());
            for (size_t out_idx = 1; out_idx < outputs.size(); ++out_idx) {
                visitor.visitDumpFeature(fnb.output(outputs[out_idx].first).buildName());
            }
        }
    }
}

bool ElementSimilarityBlueprint::setup(const fef::IIndexEnvironment& env, const fef::ParameterList& params) {
    const fef::FieldInfo* field = params[0].asField();
    _field_id = field->id();
    fef::FeatureNameBuilder fnb;
    fnb.baseName(getBaseName()).parameter(field->name());
    auto outputs = get_outputs(env.getProperties(), fnb.buildName());
    for (const auto& entry : outputs) {
        describeOutput(entry.first, entry.second);
        std::string aggr_name;
        std::string expr;
        std::string error;
        if (!vespalib::eval::Function::unwrap(entry.second, aggr_name, expr, error)) {
            LOG(warning,
                "'%s': could not extract aggregator and expression for output '%s' from config value '%s' (%s)",
                fnb.buildName().c_str(), entry.first.c_str(), entry.second.c_str(), error.c_str());
            return false;
        }
        Aggregator::UP aggr = create_aggregator(aggr_name);
        if (aggr.get() == nullptr) {
            LOG(warning, "'%s': unknown aggregator '%s'", fnb.buildName().c_str(), aggr_name.c_str());
            return false;
        }
        auto function = vespalib::eval::Function::parse(ElementSimilarityCalculator::expression_params(), expr);
        if (function->has_error()) {
            LOG(warning, "'%s': per-element expression parse error: %s", fnb.buildName().c_str(),
                function->get_error().c_str());
            return false;
        }
        _outputs.push_back(OutputContext_UP(new OutputContext(*function, std::move(aggr))));
    }
    return true;
}

fef::FeatureExecutor& ElementSimilarityBlueprint::createExecutor(const fef::IQueryEnvironment& env,
                                                                 vespalib::Stash&              stash) const {
    std::vector<OutputSpec> output_specs;
    for (const auto& output : _outputs) {
        output_specs.emplace_back(output->compile_token->get().get_function<5>(),
                                  output->aggregator_factory->create());
    }
    return stash.create<ElementSimilarityExecutor>(env, _field_id, std::move(output_specs));
}

} // namespace search::features
