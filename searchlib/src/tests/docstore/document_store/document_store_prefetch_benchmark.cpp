// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

/*
 * Benchmark comparing ways of getting the documents for the hits of a docsum request out of a document store
 * that is read with direct io and has no cache, i.e. where every document must be read from disk:
 *   read:     one document at a time
 *   sync:     as a batch, one chunk at a time
 *   io_uring: as a batch, with several chunks being read at the same time
 *
 * Usage: searchlib_document_store_prefetch_benchmark_app <directory> [num-docs] [doc-bytes] [num-queries]
 * The document store is created in the directory if it is empty, otherwise the existing one is used.
 */

#include <vespa/document/datatype/documenttype.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/fieldvalue/stringfieldvalue.h>
#include <vespa/document/repo/documenttyperepo.h>
#include <vespa/document/repo/newconfigbuilder.h>
#include <vespa/searchlib/docstore/logdocumentstore.h>
#include <vespa/searchlib/index/dummyfileheadercontext.h>
#include <vespa/searchlib/transactionlog/nosyncproxy.h>
#include <vespa/vespalib/io/async_file_reader.h>
#include <vespa/vespalib/util/threadstackexecutor.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace search;
using document::Document;
using document::DocumentTypeRepo;
using vespalib::compression::CompressionConfig;
using PrefetchMode = DocumentStore::Config::PrefetchMode;
using LidVector = IDocumentStore::LidVector;

namespace {

LogDocumentStore::Config make_config(PrefetchMode mode, uint32_t depth) {
    // No cache, such that every document must be read from disk
    return {DocumentStore::Config(CompressionConfig::NONE, 0).prefetch(mode, depth), LogDataStore::Config()};
}

std::unique_ptr<Document> make_doc(const DocumentTypeRepo& repo, uint32_t lid, size_t doc_bytes, std::mt19937& gen) {
    auto        doc = std::make_unique<Document>(repo, *repo.getDocumentType("test"),
                                          document::DocumentId("id:test:test::" + std::to_string(lid)));
    std::string text;
    while (text.size() < doc_bytes) {
        text += std::to_string(gen() % 100000);
        text.push_back(' ');
    }
    doc->setValue("main", document::StringFieldValue::make(text));
    return doc;
}

struct Result {
    std::vector<double> latencies_ms;
    size_t              docs = 0;
    uint64_t            reads = 0;
    uint64_t            direct_reads = 0;
};

size_t read_one_by_one(const IDocumentStore& store, const DocumentTypeRepo& repo, const LidVector& lids) {
    size_t docs = 0;
    for (uint32_t lid : lids) {
        docs += store.read(lid, repo) ? 1 : 0;
    }
    return docs;
}

size_t read_as_batch(const IDocumentStore& store, const DocumentTypeRepo& repo, const LidVector& lids) {
    auto prefetch = store.start_prefetch(lids);
    if (!prefetch) {
        return read_one_by_one(store, repo, lids);
    }
    size_t docs = 0;
    for (;;) {
        LidVector fetched = prefetch->fetch_more(vespalib::steady_time::max());
        if (fetched.empty()) {
            break;
        }
        for (uint32_t lid : fetched) {
            docs += prefetch->take(lid, repo) ? 1 : 0;
        }
    }
    return docs;
}

Result run(const IDocumentStore& store, const DocumentTypeRepo& repo, const std::vector<LidVector>& queries,
           bool as_batch) {
    Result result;
    auto   stats_before = store.getPrefetchStats();
    for (const LidVector& lids : queries) {
        auto start = vespalib::steady_clock::now();
        result.docs += as_batch ? read_as_batch(store, repo, lids) : read_one_by_one(store, repo, lids);
        result.latencies_ms.push_back(vespalib::count_ns(vespalib::steady_clock::now() - start) / 1000000.0);
    }
    auto stats_after = store.getPrefetchStats();
    result.reads = stats_after.reads - stats_before.reads;
    result.direct_reads = stats_after.direct_reads - stats_before.direct_reads;
    std::sort(result.latencies_ms.begin(), result.latencies_ms.end());
    return result;
}

void report(const char* distribution, uint32_t hits, const char* mode, const Result& result) {
    const auto& lat = result.latencies_ms;
    double      sum = 0;
    for (double v : lat) {
        sum += v;
    }
    printf("%-10s %5u %-12s %9.3f %9.3f %9.3f %9.3f %10.1f %10.1f %8zu\n", distribution, hits, mode,
           sum / lat.size(), lat[lat.size() / 2], lat[(lat.size() * 99) / 100], lat.back(),
           double(result.reads) / lat.size(), double(result.direct_reads) / lat.size(), result.docs);
    fflush(stdout);
}

// Hits spread evenly over the whole store
LidVector random_lids(uint32_t hits, uint32_t doc_id_limit, std::mt19937& gen) {
    LidVector lids;
    for (uint32_t i = 0; i < hits; ++i) {
        lids.push_back(1 + gen() % (doc_id_limit - 1));
    }
    return lids;
}

// Hits coming in groups of documents stored close to each other, as when they are from the same bucket
LidVector clustered_lids(uint32_t hits, uint32_t doc_id_limit, std::mt19937& gen) {
    constexpr uint32_t cluster_size = 8;
    constexpr uint32_t cluster_span = 32;
    LidVector          lids;
    while (lids.size() < hits) {
        uint32_t center = 1 + gen() % (doc_id_limit - 1 - cluster_span);
        for (uint32_t i = 0; (i < cluster_size) && (lids.size() < hits); ++i) {
            lids.push_back(center + gen() % cluster_span);
        }
    }
    std::shuffle(lids.begin(), lids.end(), gen);
    return lids;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <directory> [num-docs] [doc-bytes] [num-queries]\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    uint32_t    num_docs = (argc > 2) ? atoi(argv[2]) : 300000;
    size_t      doc_bytes = (argc > 3) ? atoi(argv[3]) : 2000;
    uint32_t    num_queries = (argc > 4) ? atoi(argv[4]) : 300;
    uint32_t    depth = 32;

    document::new_config_builder::NewConfigBuilder builder;
    builder.document("test", 4711).addField("main", builder.stringTypeRef());
    DocumentTypeRepo repo(builder.config());

    index::DummyFileHeaderContext file_header_context;
    vespalib::ThreadStackExecutor executor(1);
    transactionlog::NoSyncProxy   tl_syncer;
    TuneFileSummary               tune;
    tune._randRead.setWantDirectIO();

    std::filesystem::create_directories(dir);
    if (std::filesystem::is_empty(dir)) {
        printf("Creating document store with %u documents of about %zu bytes in %s\n", num_docs, doc_bytes,
               dir.c_str());
        LogDocumentStore store(executor, dir, make_config(PrefetchMode::OFF, 1), GrowStrategy(), tune,
                               file_header_context, tl_syncer, nullptr);
        std::mt19937     gen(42);
        for (uint32_t lid = 1; lid <= num_docs; ++lid) {
            store.write(lid, lid, *make_doc(repo, lid, doc_bytes, gen));
        }
        store.flush(store.initFlush(num_docs));
    }
    // All files written by a previous instance are read only, and read directly from disk
    LogDocumentStore store(executor, dir, make_config(PrefetchMode::OFF, 1), GrowStrategy(), tune,
                           file_header_context, tl_syncer, nullptr);
    uint32_t doc_id_limit = store.getDocIdLimit();
    printf("Document store has %u documents using %zu bytes on disk, io_uring %s, %u queries per case\n",
           doc_id_limit - 1, store.getDiskFootprint(),
           vespalib::AsyncFileReader::io_uring_available() ? "available" : "NOT available (blocking reads used)",
           num_queries);
    printf("%-10s %5s %-12s %9s %9s %9s %9s %10s %10s %8s\n", "lids", "hits", "mode", "avg ms", "p50 ms", "p99 ms",
           "max ms", "reads/q", "direct/q", "docs");
    for (const char* distribution : {"random", "clustered"}) {
        for (uint32_t hits : {10, 50, 200}) {
            std::mt19937           gen(hits);
            std::vector<LidVector> queries;
            for (uint32_t i = 0; i < num_queries; ++i) {
                queries.push_back((distribution == std::string("random")) ? random_lids(hits, doc_id_limit, gen)
                                                                          : clustered_lids(hits, doc_id_limit, gen));
            }
            store.reconfigure(make_config(PrefetchMode::OFF, 1));
            report(distribution, hits, "read", run(store, repo, queries, false));
            store.reconfigure(make_config(PrefetchMode::SYNC, 1));
            report(distribution, hits, "sync", run(store, repo, queries, true));
            store.reconfigure(make_config(PrefetchMode::IO_URING, depth));
            report(distribution, hits, "io_uring", run(store, repo, queries, true));
        }
    }
    return 0;
}
