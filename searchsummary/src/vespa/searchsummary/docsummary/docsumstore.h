// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace search::docsummary {

class IDocsumStoreDocument;

/**
 * Interface used to fetch docsum specific abstract of documents based on local document id.
 **/
class IDocsumStore {
public:
    using UP = std::unique_ptr<IDocsumStore>;

    virtual ~IDocsumStore() = default;

    /**
     * Get a docsum specific abstract of the document for the given local document id.
     **/
    virtual std::unique_ptr<const IDocsumStoreDocument> get_document(uint32_t docid) = 0;

    /**
     * Start fetching the documents for the given local document ids as a batch, for those that are
     * expensive to get one by one. Must be called before get_document() to have any effect.
     * @return the local document ids being fetched. To benefit from the batching, get_document()
     *         should not be called for these until they have been returned by fetch_more().
     **/
    virtual std::vector<uint32_t> start_prefetch(const std::vector<uint32_t>& docids) {
        (void)docids;
        return {};
    }

    /**
     * Wait for more of the documents being fetched as a batch.
     * @return the local document ids that are now cheap to get, empty when there are no more to wait for
     *         or the deadline expired first.
     **/
    virtual std::vector<uint32_t> fetch_more(std::chrono::steady_clock::time_point deadline) {
        (void)deadline;
        return {};
    }
};

} // namespace search::docsummary
