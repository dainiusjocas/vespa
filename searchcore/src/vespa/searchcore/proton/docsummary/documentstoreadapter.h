// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/searchlib/docstore/idocumentstore.h>
#include <vespa/searchsummary/docsummary/docsumstore.h>

namespace proton {

class DocumentStoreAdapter : public search::docsummary::IDocsumStore {
private:
    const search::IDocumentStore&                   _docStore;
    const document::DocumentTypeRepo&               _repo;
    std::unique_ptr<search::IDocumentStorePrefetch> _prefetch;

public:
    DocumentStoreAdapter(const search::IDocumentStore& docStore, const document::DocumentTypeRepo& repo);
    ~DocumentStoreAdapter() override;

    std::unique_ptr<const search::docsummary::IDocsumStoreDocument> get_document(uint32_t docId) override;
    std::vector<uint32_t> start_prefetch(const std::vector<uint32_t>& docids) override;
    std::vector<uint32_t> fetch_more(vespalib::steady_time deadline) override;
};

} // namespace proton
