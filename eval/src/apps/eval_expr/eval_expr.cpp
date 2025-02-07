// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/util/require.h>
#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/value_type.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/lazy_params.h>
#include <vespa/eval/eval/interpreted_function.h>
#include <vespa/eval/eval/feature_name_extractor.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/make_tensor_function.h>
#include <vespa/eval/eval/optimize_tensor_function.h>
#include <vespa/eval/eval/compile_tensor_function.h>
#include <vespa/eval/eval/test/test_io.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/util/time.h>
#include <vespa/vespalib/io/mapped_file_input.h>
#include <cctype>

#include <histedit.h>

using vespalib::make_string_short::fmt;

using namespace vespalib::eval;
using namespace vespalib::eval::test;

using vespalib::Slime;
using vespalib::slime::JsonFormat;
using vespalib::slime::Inspector;
using vespalib::slime::Cursor;
using vespalib::Input;
using vespalib::Memory;

using CostProfile = std::vector<std::pair<size_t,vespalib::duration>>;

const auto &factory = FastValueBuilderFactory::get();

void list_commands(FILE *file, const char *prefix) {
    fprintf(file, "%s'exit' -> exit the program\n", prefix);
    fprintf(file, "%s'help' -> print available commands\n", prefix);
    fprintf(file, "%s'list' -> list named values\n", prefix);
    fprintf(file, "%s'verbose (true|false)' -> enable or disable verbose output\n", prefix);
    fprintf(file, "%s'def <name> <expr>' -> evaluate expression, bind result to a name\n", prefix);
    fprintf(file, "%s'undef <name>' -> remove a named value\n", prefix);    
    fprintf(file, "%s'<expr>' -> evaluate expression\n", prefix);
}

int usage(const char *self) {
    //               -------------------------------------------------------------------------------
    fprintf(stderr, "usage: %s [--verbose] <expr> [expr ...]\n", self);
    fprintf(stderr, "  Evaluate a sequence of expressions. The first expression must be\n");
    fprintf(stderr, "  self-contained (no external values). Later expressions may use the\n");
    fprintf(stderr, "  results of earlier expressions. Expressions are automatically named\n");
    fprintf(stderr, "  using single letter symbols ('a' through 'z'). Quote expressions to\n");
    fprintf(stderr, "  make sure they become separate parameters. The --verbose option may\n");
    fprintf(stderr, "  be specified to get more detailed informaion about how the various\n");
    fprintf(stderr, "  expressions are optimized and executed.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "example: %s \"2+2\" \"a+2\" \"a+b\"\n", self);
    fprintf(stderr, "  (a=4, b=6, c=10)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "advanced usage: %s interactive\n", self);
    fprintf(stderr, "  This runs the progam in interactive mode. possible commands (line based):\n");
    list_commands(stderr, "    ");
    fprintf(stderr, "\n");
    fprintf(stderr, "advanced usage: %s json-repl\n", self);
    fprintf(stderr, "  This will put the program into a read-eval-print loop where it reads\n");
    fprintf(stderr, "  json objects from stdin and writes json objects to stdout.\n");
    fprintf(stderr, "  possible commands: (object based)\n");
    fprintf(stderr, "    {expr:<expr>, ?name:<name>, ?verbose:true}\n");
    fprintf(stderr, "    -> { result:<verbatim-expr> ?steps:[{class:string,symbol:string}] }\n");
    fprintf(stderr, "      Evaluate an expression and return the result. If a name is specified,\n");
    fprintf(stderr, "      the result will be bound to that name and will be available as a symbol\n");
    fprintf(stderr, "      when doing future evaluations. Verbose output must be enabled for each\n");
    fprintf(stderr, "      relevant command and will result in the 'steps' field being populated in\n");
    fprintf(stderr, "      the response.\n");
    fprintf(stderr, "  if any command fails, the response will be { error:string }\n");
    fprintf(stderr, "  commands may be batched using json arrays:\n");
    fprintf(stderr, "    [cmd1,cmd2,cmd3] -> [res1,res2,res3]\n");
    fprintf(stderr, "\n");
    //               -------------------------------------------------------------------------------
    return 1;
}

int overflow(int cnt, int max) {
    fprintf(stderr, "error: too many expressions: %d (max is %d)\n", cnt, max);
    return 2;
}

class Context
{
private:
    std::vector<std::string> _param_names;
    std::vector<ValueType>   _param_types;
    std::vector<Value::UP>   _param_values;
    std::vector<Value::CREF> _param_refs;
    bool                     _verbose;
    std::string              _error;
    CTFMetaData              _meta;
    CostProfile              _cost;

    void clear_state() {
        _error.clear();
        _meta = CTFMetaData();
        _cost.clear();
    }

public:
    Context() : _param_names(), _param_types(), _param_values(), _param_refs(), _verbose(), _meta(), _cost() {}
    ~Context();

    void verbose(bool value) { _verbose = value; }
    bool verbose() const { return _verbose; }

    size_t size() const { return _param_names.size(); }
    const std::string &name(size_t idx) const { return _param_names[idx]; }
    const ValueType &type(size_t idx) const { return _param_types[idx]; }

    Value::UP eval(const std::string &expr) {
        clear_state();
        SimpleObjectParams params(_param_refs);
        auto fun = Function::parse(_param_names, expr, FeatureNameExtractor());
        if (fun->has_error()) {
            _error = fmt("expression parsing failed: %s", fun->get_error().c_str());
            return {};
        }
        NodeTypes types = NodeTypes(*fun, _param_types);
        ValueType res_type = types.get_type(fun->root());
        if (res_type.is_error() || !types.errors().empty()) {
            _error = fmt("type resolving failed for expression: '%s'", expr.c_str());
            for (const auto &issue: types.errors()) {
                _error.append(fmt("\n  type issue: %s", issue.c_str()));
            }
            return {};
        }
        vespalib::Stash stash;
        const TensorFunction &plain_fun = make_tensor_function(factory, fun->root(), types, stash);
        const TensorFunction &optimized = optimize_tensor_function(factory, plain_fun, stash);
        Value::UP result;
        if (_verbose) {
            InterpretedFunction ifun(factory, optimized, &_meta);
            REQUIRE_EQ(_meta.steps.size(), ifun.program_size());
            InterpretedFunction::ProfiledContext ctx(ifun);
            result = factory.copy(ifun.eval(ctx, params));
            _cost = ctx.cost;
        } else {
            InterpretedFunction ifun(factory, optimized, nullptr);
            InterpretedFunction::Context ctx(ifun);
            result = factory.copy(ifun.eval(ctx, params));
        }
        REQUIRE_EQ(result->type(), res_type);
        return result;
    }

    const std::string &error() const { return _error; }
    const CTFMetaData &meta() const { return _meta; }
    const CostProfile &cost() const { return _cost; }

    bool save(const std::string &name, Value::UP value) {
        REQUIRE(value);
        for (size_t i = 0; i < _param_names.size(); ++i) {
            if (_param_names[i] == name) {
                _param_types[i] = value->type();
                _param_values[i] = std::move(value);
                _param_refs[i] = *_param_values[i];
                return true;
            }
        }
        _param_names.push_back(name);
        _param_types.push_back(value->type());
        _param_values.push_back(std::move(value));
        _param_refs.emplace_back(*_param_values.back());
        return false;
    }

    bool remove(const std::string &name) {
        for (size_t i = 0; i < _param_names.size(); ++i) {
            if (_param_names[i] == name) {
                _param_names.erase(_param_names.begin() + i);
                _param_types.erase(_param_types.begin() + i);
                _param_values.erase(_param_values.begin() + i);
                _param_refs.erase(_param_refs.begin() + i);
                return true;
            }
        }
        return false;
    }
};
Context::~Context() = default;

void print_error(const std::string &error) {
    fprintf(stderr, "error: %s\n", error.c_str());
}

void print_value(const Value &value, const std::string &name, const CTFMetaData &meta, const CostProfile &cost) {
    bool with_name = !name.empty();
    bool with_meta = !meta.steps.empty();
    auto spec = spec_from_value(value);
    if (with_meta) {
        if (with_name) {
            fprintf(stderr, "meta-data(%s):\n", name.c_str());
        } else {
            fprintf(stderr, "meta-data:\n");
        }
        const auto &steps = meta.steps;
        for (size_t i = 0; i < steps.size(); ++i) {
            fprintf(stderr, "  class: %s\n", steps[i].class_name.c_str());
            fprintf(stderr, "    symbol: %s\n", steps[i].symbol_name.c_str());
            fprintf(stderr, "    count: %zu\n", cost[i].first);
            fprintf(stderr, "    time_us: %g\n", vespalib::count_ns(cost[i].second)/1000.0);
        }
    }
    if (with_name) {
        fprintf(stdout, "%s: ", name.c_str());
    }
    if (value.type().is_double()) {
        fprintf(stdout, "%.32g\n", spec.as_double());
    } else {
        fprintf(stdout, "%s\n", spec.to_string().c_str());
    }
}

void handle_message(Context &ctx, const Inspector &req, Cursor &reply) {
    std::string expr = req["expr"].asString().make_string();
    std::string name = req["name"].asString().make_string();
    ctx.verbose(req["verbose"].asBool());
    if (expr.empty()) {
        reply.setString("error", "missing expression (field name: 'expr')");
        return;
    }
    auto value = ctx.eval(expr);
    if (!value) {
        reply.setString("error", ctx.error());
        return;
    }
    reply.setString("result", spec_from_value(*value).to_expr());
    if (!name.empty()) {
        ctx.save(name, std::move(value));
    }
    if (!ctx.meta().steps.empty()) {
        auto &steps_out = reply.setArray("steps");
        for (const auto &step: ctx.meta().steps) {
            auto &step_out = steps_out.addObject();
            step_out.setString("class", step.class_name);
            step_out.setString("symbol", step.symbol_name);            
        }
    }
}

bool is_hash_bang(const std::string &str) {
    if (str.size() > 2) {
        return str[0] == '#' && str[1] == '!';
    }
    return false;
}

bool is_only_whitespace(const std::string &str) {
    for (auto c : str) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

class Script {
private:
    std::unique_ptr<Input> _input;
    LineReader _reader;
    bool _script_only = false;
public:
    Script(std::unique_ptr<Input> input)
      : _input(std::move(input)), _reader(*_input) {}
    static auto empty() {
        struct EmptyInput : Input {
            Memory obtain() override { return Memory(); }
            Input &evict(size_t) override { return *this; }
        };
        return std::make_unique<Script>(std::make_unique<EmptyInput>());
    }
    static auto from_file(const std::string &file_name) {
        auto input = std::make_unique<vespalib::MappedFileInput>(file_name);
        if (!input->valid()) {
            fprintf(stderr, "warning: could not read script: %s\n", file_name.c_str());
        }
        return std::make_unique<Script>(std::move(input));
    }
    Script &script_only(bool value) {
        _script_only = value;
        return *this;
    }
    bool script_only() const { return _script_only; }
    bool read_line(std::string &line) { return _reader.read_line(line); }
};

class Collector {
private:
    Slime _slime;
    Cursor &_obj;
    Cursor &_arr;
    bool _enabled;
    std::string _error;
public:
    Collector()
      : _slime(), _obj(_slime.setObject()), _arr(_obj.setArray("f")), _enabled(false) {}
    void enable() {
        _enabled = true;
    }
    void fail(const std::string &msg) {
        if (_error.empty()) {
            _error = msg;
        }
    }
    const std::string &error() const {
        return _error;
    }
    void comment(const std::string &text) {
        if (_enabled) {
            Cursor &f = _arr.addObject();
            f.setString("op", "c");
            Cursor &p = f.setObject("p");
            p.setString("t", text);
        }
    }
    void expr(const std::string &name, const std::string &expr) {
        if (_enabled) {
            Cursor &f = _arr.addObject();
            f.setString("op", "e");
            Cursor &p = f.setObject("p");
            p.setString("n", name);
            p.setString("e", expr);
        }
    }
    std::string toString() const {
        return _slime.toString();
    }
};

struct EditLineWrapper {
    EditLine *my_el;
    History *my_hist;
    HistEvent ignore;
    Script &script;
    static std::string prompt;
    static char *prompt_fun(EditLine *) { return &prompt[0]; }
    EditLineWrapper(Script &script_in)
      : my_el(el_init("vespa-eval-expr", stdin, stdout, stderr)),
        my_hist(history_init()),
        script(script_in)
    {
        memset(&ignore, 0, sizeof(ignore));
        el_set(my_el, EL_EDITOR, "emacs");
        el_set(my_el, EL_PROMPT, prompt_fun);
        history(my_hist, &ignore, H_SETSIZE, 1024);
        el_set(my_el, EL_HIST, history, my_hist);
    }
    ~EditLineWrapper();
    bool read_line(std::string &line_out) {
        bool from_script = false;
        do {
            from_script = script.read_line(line_out);
            if (!from_script) {
                if (script.script_only()) {
                    return false;
                }
                int line_len = 0;
                const char *line = el_gets(my_el, &line_len);
                if (line == nullptr) {
                    return false;
                }
                line_out.assign(line, line_len);
            }
            if ((line_out.size() > 0) && (line_out[line_out.size() - 1] == '\n')) {
                line_out.pop_back();
            }
        } while (is_hash_bang(line_out) || is_only_whitespace(line_out));
        if (from_script) {
            fprintf(stdout, "%s%s\n", prompt.c_str(), line_out.c_str());
        }
        history(my_hist, &ignore, H_ENTER, line_out.c_str());
        return true;
    }
};
EditLineWrapper::~EditLineWrapper()
{
    el_set(my_el, EL_HIST, history, nullptr);
    history_end(my_hist);
    el_end(my_el);
}
std::string EditLineWrapper::prompt("> ");

const std::string exit_cmd("exit");
const std::string help_cmd("help");
const std::string list_cmd("list");
const std::string verbose_cmd("verbose ");
const std::string def_cmd("def ");
const std::string undef_cmd("undef ");
const std::string ignore_cmd("#");

int interactive_mode(Context &ctx, Script &script, Collector &collector) {
    EditLineWrapper input(script);
    std::string line;
    while (input.read_line(line)) {
        if (line == exit_cmd) {
            return 0;
        }
        if (line == help_cmd) {
            list_commands(stdout, "  ");
            continue;
        }
        if (line == list_cmd) {
            for (size_t i = 0; i < ctx.size(); ++i) {
                fprintf(stdout, "  %s: %s\n", ctx.name(i).c_str(), ctx.type(i).to_spec().c_str());
            }
            continue;
        }
        if (line.find(ignore_cmd) == 0) {
            collector.comment(line.substr(ignore_cmd.size()));
            continue;
        }
        if (line.find(verbose_cmd) == 0) {
            auto flag_str = line.substr(verbose_cmd.size());
            bool flag = (flag_str == "true");
            bool bad = (!flag && (flag_str != "false"));
            if (bad) {
                fprintf(stderr, "bad flag specifier: '%s', must be 'true' or 'false'\n", flag_str.c_str());                
            } else {
                ctx.verbose(flag);
                fprintf(stdout, "verbose set to %s\n", flag ? "true" : "false");
            }
            continue;
        }
        std::string name;
        if (line.find(undef_cmd) == 0) {
            name = line.substr(undef_cmd.size());
            if (ctx.remove(name)) {
                fprintf(stdout, "removed value '%s'\n", name.c_str());
            } else {
                fprintf(stdout, "value not found: '%s'\n", name.c_str());
            }
            collector.fail("undef operation not supported");
            continue;
        }
        std::string expr;
        if (line.find(def_cmd) == 0) {
            auto name_size = (line.find(" ", def_cmd.size()) - def_cmd.size());
            name = line.substr(def_cmd.size(), name_size);
            expr = line.substr(def_cmd.size() + name_size + 1);
        } else {
            expr = line;
        }
        if (ctx.verbose()) {
            fprintf(stderr, "eval '%s'", expr.c_str());
            if (name.empty()) {
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, " -> '%s'\n", name.c_str());
            }
        }
        collector.expr(name, expr);
        if (auto value = ctx.eval(expr)) {
            print_value(*value, name, ctx.meta(), ctx.cost());
            if (!name.empty()) {
                if (ctx.save(name, std::move(value))) {
                    collector.fail("value redefinition not supported");
                }
            }
        } else {
            collector.fail("sub-expression evaluation failed");
            print_error(ctx.error());
        }
    }
    return 0;
}

int json_repl_mode(Context &ctx) {
    StdIn std_in;
    StdOut std_out;
    for (;;) {
        if (look_for_eof(std_in)) {
            return 0;
        }
        Slime req;
        if (!JsonFormat::decode(std_in, req)) {
            return 3;
        }
        Slime reply;
        if (req.get().type().getId() == vespalib::slime::ARRAY::ID) {
            reply.setArray();
            for (size_t i = 0; i < req.get().entries(); ++i) {
                handle_message(ctx, req[i], reply.get().addObject());
            }
        } else {
            handle_message(ctx, req.get(), reply.setObject());
        }
        write_compact(reply, std_out);
    }
}

int main(int argc, char **argv) {
    bool verbose = ((argc > 1) && (std::string(argv[1]) == "--verbose"));
    int expr_idx = verbose ? 2 : 1;
    int expr_cnt = (argc - expr_idx);
    int expr_max = ('z' - 'a') + 1;
    if (expr_cnt == 0) {
        return usage(argv[0]);
    }
    if (expr_cnt > expr_max) {
        return overflow(expr_cnt, expr_max);
    }
    Context ctx;
    if ((expr_cnt == 1) && (std::string(argv[expr_idx]) == "interactive")) {
        setlocale(LC_ALL, "");
        Collector ignored;
        return interactive_mode(ctx, *Script::empty(), ignored);
    }
    if ((expr_cnt == 2) && (std::string(argv[expr_idx]) == "interactive")) {
        setlocale(LC_ALL, "");
        Collector ignored;
        return interactive_mode(ctx, *Script::from_file(argv[expr_idx + 1]), ignored);
    }
    if ((expr_cnt == 3) &&
        (std::string(argv[expr_idx]) == "interactive") &&
        (std::string(argv[expr_idx + 2]) == "convert"))
    {
        setlocale(LC_ALL, "");
        Collector collector;
        collector.enable();
        interactive_mode(ctx, Script::from_file(argv[expr_idx + 1])->script_only(true), collector);
        if (collector.error().empty()) {
            fprintf(stdout, "%s\n", collector.toString().c_str());
            return 0;
        } else {
            fprintf(stderr, "conversion failed: %s\n", collector.error().c_str());
            return 3;
        }
    }
    if ((expr_cnt == 1) && (std::string(argv[expr_idx]) == "json-repl")) {
        return json_repl_mode(ctx);
    }
    ctx.verbose(verbose);
    std::string name("a");
    for (int i = expr_idx; i < argc; ++i) {
        if (auto value = ctx.eval(argv[i])) {
            if (expr_cnt > 1) {
                print_value(*value, name, ctx.meta(), ctx.cost());
                ctx.save(name, std::move(value));
                ++name[0];
            } else {
                std::string no_name;
                print_value(*value, no_name, ctx.meta(), ctx.cost());
            }
        } else {
            print_error(ctx.error());
            return 3;
        }
    }
    return 0;
}
