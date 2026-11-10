// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "docsumcontext.h"

#include <vespa/check_require.h>
#include <vespa/document/datatype/positiondatatype.h>
#include <vespa/searchcore/proton/matching/matcher.h>
#include <vespa/searchlib/attribute/iattributemanager.h>
#include <vespa/searchlib/common/location.h>
#include <vespa/searchlib/common/matching_elements.h>
#include <vespa/searchlib/queryeval/begin_and_end_id.h>
#include <vespa/vespalib/data/slime/slime.h>
#include <vespa/vespalib/util/stringfmt.h>

#include <algorithm>

#include <vespa/log/log.h>
LOG_SETUP(".proton.docsummary.docsumcontext");

using document::PositionDataType;
using search::common::Location;
using std::string;
using vespalib::make_string;
using vespalib::Memory;
using vespalib::Slime;
using vespalib::slime::Cursor;
using vespalib::slime::Inserter;
using vespalib::slime::NIX;
using vespalib::slime::ObjectSymbolInserter;
using vespalib::slime::Symbol;
using vespalib::slime::SymbolTable;
using namespace search;
using namespace search::attribute;
using namespace search::engine;
using namespace search::docsummary;

namespace proton {

using namespace matching;

namespace {

Memory DOCSUMS("docsums");
Memory DOCSUM("docsum");
Memory ERRORS("errors");
Memory TYPE("type");
Memory MESSAGE("message");
Memory TIMEOUT("timeout");

} // namespace

void DocsumContext::initState() {
    _docsumState.query_normalization(this);
    const DocsumRequest& req = _request;
    _docsumState._args.initFromDocsumRequest(req);
    auto [session, expectSession] = Matcher::lookupSearchSession(_sessionMgr, req);
    if (session) {
        _docsumState._args.setSerializedQueryTree(session->getSerializedQueryTree());
    }
    _docsumState._docsumbuf.clear();
    _docsumState._docsumbuf.reserve(req.hits.size());
    for (const auto& hit : req.hits) {
        _docsumState._docsumbuf.push_back(hit.docid);
    }
}

vespalib::Slime::UP DocsumContext::createSlimeReply() {
    IDocsumWriter::ResolveClassInfo rci =
        _docsumWriter.resolveClassInfo(_docsumState._args.getResultClassName(), _docsumState._args.get_fields());
    _docsumWriter.initState(_attrMgr, _docsumState, rci);
    const size_t estimatedChunkSize(std::min(0x200000ul, _docsumState._docsumbuf.size() * 0x400ul));
    auto         response = std::make_unique<vespalib::Slime>(Slime::Params(estimatedChunkSize));
    Cursor&      root = response->setObject();
    Cursor&      array = root.setArray(DOCSUMS);
    const Symbol docsumSym = response->insert(DOCSUM);
    _docsumState._omit_summary_features = (rci.res_class == nullptr) || rci.res_class->omit_summary_features();
    const std::vector<uint32_t>& docids = _docsumState._docsumbuf;
    auto                         insert_docsum = [&](Cursor& docSumC, uint32_t docId) {
        ObjectSymbolInserter inserter(docSumC, docsumSym);
        if ((docId != search::endDocId) && rci.res_class != nullptr) {
            _docsumWriter.insertDocsum(rci, docId, _docsumState, _docsumStore, inserter);
        }
    };
    std::vector<uint32_t> prefetched;
    if ((rci.res_class != nullptr) && !rci.all_fields_generated) {
        prefetched = _docsumStore.start_prefetch(docids);
    }
    uint32_t num_ok(0);
    if (prefetched.empty()) {
        for (uint32_t docId : docids) {
            if (_request.expired()) {
                break;
            }
            insert_docsum(array.addObject(), docId);
            num_ok++;
        }
    } else {
        // The hits not waiting for a document to be fetched are filled first, then the rest as their
        // documents arrive. As hits are not filled in order, all entries must exist up front.
        std::sort(prefetched.begin(), prefetched.end());
        std::vector<Cursor*>                       entries;
        std::vector<std::pair<uint32_t, uint32_t>> waiting; // docid, index of hit
        entries.reserve(docids.size());
        for (uint32_t i = 0; i < docids.size(); ++i) {
            entries.push_back(&array.addObject());
            if (std::binary_search(prefetched.begin(), prefetched.end(), docids[i])) {
                waiting.emplace_back(docids[i], i);
            }
        }
        std::sort(waiting.begin(), waiting.end());
        for (uint32_t i = 0; (i < docids.size()) && !_request.expired(); ++i) {
            if (!std::binary_search(prefetched.begin(), prefetched.end(), docids[i])) {
                insert_docsum(*entries[i], docids[i]);
                num_ok++;
            }
        }
        uint32_t num_batches = 0;
        while (!_request.expired()) {
            std::vector<uint32_t> fetched = _docsumStore.fetch_more(_request.getTimeOfDoom());
            if (fetched.empty()) {
                break;
            }
            ++num_batches;
            for (uint32_t docId : fetched) {
                auto hit = std::lower_bound(waiting.begin(), waiting.end(), std::make_pair(docId, 0u));
                for (; (hit != waiting.end()) && (hit->first == docId) && !_request.expired(); ++hit) {
                    insert_docsum(*entries[hit->second], docId);
                    num_ok++;
                }
            }
        }
        if (_request.trace().shouldTrace(5)) {
            _request.trace().addEvent(5, make_string("Summary prefetch: %zu of %zu hits waited for %zu documents, "
                                                     "fetched in %u batches",
                                                     waiting.size(), docids.size(), prefetched.size(), num_batches));
        }
    }
    if (num_ok != docids.size()) {
        const uint32_t numTimedOut = docids.size() - num_ok;
        Cursor&        errors = root.setArray(ERRORS);
        Cursor&        timeout = errors.addObject();
        timeout.setString(TYPE, TIMEOUT);
        timeout.setString(MESSAGE, make_string("Timed out %d summaries with %" PRId64 "us left.", numTimedOut,
                                               vespalib::count_us(_request.getTimeLeft())));
    }
    return response;
}

DocsumContext::DocsumContext(const DocsumRequest& request, IDocsumWriter& docsumWriter, IDocsumStore& docsumStore,
                             std::shared_ptr<Matcher> matcher, ISearchContext& searchCtx, IAttributeContext& attrCtx,
                             const IAttributeManager& attrMgr, SessionManager& sessionMgr)
    : _request(request),
      _docsumWriter(docsumWriter),
      _docsumStore(docsumStore),
      _matcher(std::move(matcher)),
      _searchCtx(searchCtx),
      _attrCtx(attrCtx),
      _attrMgr(attrMgr),
      _docsumState(*this),
      _sessionMgr(sessionMgr) {
    initState();
}

DocsumReply::UP DocsumContext::getDocsums() {
    return std::make_unique<DocsumReply>(createSlimeReply());
}

void DocsumContext::fillSummaryFeatures(search::docsummary::GetDocsumsState& state) {
    CHECK(&_docsumState == &state);
    if (_matcher->canProduceSummaryFeatures()) {
        state._summaryFeatures = _matcher->getSummaryFeatures(_request, _searchCtx, _attrCtx, _sessionMgr);
    }
}

void DocsumContext::fillRankFeatures(search::docsummary::GetDocsumsState& state) {
    CHECK(&_docsumState == &state);
    // check if we are allowed to run
    if (!state._args.dumpFeatures()) {
        return;
    }
    state._rankFeatures = _matcher->getRankFeatures(_request, _searchCtx, _attrCtx, _sessionMgr);
}

std::unique_ptr<MatchingElements> DocsumContext::fill_matching_elements(const MatchingElementsFields& fields) {
    if (_matcher) {
        return _matcher->get_matching_elements(_request, _searchCtx, _attrCtx, _sessionMgr, fields);
    }
    return std::make_unique<MatchingElements>();
}

bool DocsumContext::is_text_matching(std::string_view) const noexcept {
    // this is for dynamic teaser only; all those fields should be text matching
    return true;
}

Normalizing DocsumContext::normalizing_mode(std::string_view) const noexcept {
    // this is for dynamic teaser only; it always does lowercase/accent removal.
    return Normalizing::LOWERCASE_AND_FOLD;
}

} // namespace proton
