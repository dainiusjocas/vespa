// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "documentstoreadapter.h"

#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/fieldvalue/stringfieldvalue.h>
#include <vespa/document/fieldvalue/tensorfieldvalue.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/searchsummary/docsummary/docsum_store_document.h>
#include <vespa/vespalib/objects/nbostream.h>

#include <vespa/log/log.h>
LOG_SETUP(".proton.docsummary.documentstoreadapter");

using namespace document;
using namespace search::docsummary;

namespace proton {

DocumentStoreAdapter::DocumentStoreAdapter(const search::IDocumentStore& docStore, const DocumentTypeRepo& repo)
    : _docStore(docStore), _repo(repo), _prefetch() {
}

DocumentStoreAdapter::~DocumentStoreAdapter() = default;

std::unique_ptr<const IDocsumStoreDocument> DocumentStoreAdapter::get_document(uint32_t docId) {
    std::unique_ptr<document::Document> document;
    if (_prefetch) {
        document = _prefetch->take(docId, _repo);
    }
    if (!document) {
        document = _docStore.read(docId, _repo);
    }
    if (!document) {
        LOG(debug, "Did not find summary document for docId %u. Returning empty docsum", docId);
        return {};
    }
    LOG(spam, "getMappedDocSum(%u): document={\n%s\n}", docId, document->toString(true).c_str());
    return std::make_unique<DocsumStoreDocument>(std::move(document));
}

std::vector<uint32_t> DocumentStoreAdapter::start_prefetch(const std::vector<uint32_t>& docids) {
    _prefetch = _docStore.start_prefetch(docids);
    return _prefetch ? _prefetch->lids() : std::vector<uint32_t>();
}

std::vector<uint32_t> DocumentStoreAdapter::fetch_more(vespalib::steady_time deadline) {
    return _prefetch ? _prefetch->fetch_more(deadline) : std::vector<uint32_t>();
}

} // namespace proton
