// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "documentstore.h"

#include "ibucketizer.h"
#include "value.h"
#include "visitcache.h"

#include <vespa/check_require.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/vespalib/data/databuffer.h>
#include <vespa/vespalib/io/async_file_reader.h>
#include <vespa/vespalib/stllike/hash_map.hpp>
#include <vespa/vespalib/util/compressor.h>
#include <vespa/vespalib/util/size_literals.h>

#include <vespa/vespalib/stllike/cache.hpp>

#include <algorithm>
#include <compare>

#include <vespa/log/log.h>

LOG_SETUP(".searchlib.docstore.documentstore");

using document::DocumentTypeRepo;
using vespalib::CacheStats;
using vespalib::compression::CompressionConfig;

namespace search {

namespace {

class DocumentVisitorAdapter : public IBufferVisitor {
public:
    DocumentVisitorAdapter(const DocumentTypeRepo& repo, IDocumentVisitor& visitor)
        : _repo(repo), _visitor(visitor) {}
    void visit(uint32_t lid, vespalib::ConstBufferRef buf) override;

private:
    const DocumentTypeRepo& _repo;
    IDocumentVisitor&       _visitor;
};

void DocumentVisitorAdapter::visit(uint32_t lid, vespalib::ConstBufferRef buf) {
    if (buf.size() > 0) {
        vespalib::nbostream is(buf.c_str(), buf.size());
        _visitor.visit(lid, std::make_unique<document::Document>(_repo, is));
    }
}

} // namespace

using vespalib::nbostream;

namespace docstore {

class BackingStore {
public:
    BackingStore(IDataStore& store, CompressionConfig compression)
        : _backingStore(store), _compression(compression) {}

    bool read(DocumentIdT key, Value& value) const;
    void visit(const IDocumentStore::LidVector& lids, const DocumentTypeRepo& repo, IDocumentVisitor& visitor) const;
    void write(DocumentIdT, const Value&);
    void erase(DocumentIdT) {}
    CompressionConfig getCompression() const { return _compression.load(std::memory_order_relaxed); }
    void reconfigure(CompressionConfig compression);

private:
    IDataStore&                    _backingStore;
    std::atomic<CompressionConfig> _compression;
};

void BackingStore::visit(const IDocumentStore::LidVector& lids, const DocumentTypeRepo& repo,
                         IDocumentVisitor& visitor) const {
    DocumentVisitorAdapter adapter(repo, visitor);
    _backingStore.read(lids, adapter);
}

bool BackingStore::read(DocumentIdT key, Value& value) const {
    bool                 found(false);
    vespalib::DataBuffer buf(4_Ki);
    ssize_t              len = _backingStore.read(key, buf);
    if (len > 0) {
        value.set(std::move(buf), len, getCompression());
        found = true;
    }
    return found;
}

void BackingStore::write(DocumentIdT lid, const Value& value) {
    Value::Result buf = value.decompressed();
    CHECK(buf.second);
    _backingStore.write(value.getSyncToken(), lid, buf.first.getData(), buf.first.getDataLen());
}

void BackingStore::reconfigure(CompressionConfig compression) {
    _compression.store(compression, std::memory_order_relaxed);
}

using CacheParams = vespalib::CacheParam<vespalib::LruParam<DocumentIdT, docstore::Value>, docstore::BackingStore,
                                         vespalib::zero<DocumentIdT>, vespalib::size<docstore::Value>>;

class Cache : public vespalib::cache<CacheParams> {
public:
    Cache(BackingStore& b, size_t maxBytes) : vespalib::cache<CacheParams>(b, maxBytes) {}
};

} // namespace docstore

using docstore::Value;

namespace {

uint32_t prefetch_max_in_flight(const DocumentStore::Config& config) {
    using PrefetchMode = DocumentStore::Config::PrefetchMode;
    switch (config.prefetchMode()) {
    case PrefetchMode::OFF:
        return 0;
    case PrefetchMode::SYNC:
        return 1;
    case PrefetchMode::IO_URING:
        (void)vespalib::AsyncFileReader::io_uring_available(); // Logs whether it is, before it is needed
        return std::max(2u, config.prefetchDepth());
    }
    return 0;
}

} // namespace

bool DocumentStore::Config::operator==(const Config& rhs) const {
    return (_maxCacheBytes == rhs._maxCacheBytes) && (_updateStrategy == rhs._updateStrategy) &&
           (_compression == rhs._compression) && (_prefetchMode == rhs._prefetchMode) &&
           (_prefetchDepth == rhs._prefetchDepth);
}

size_t DocumentStore::getCacheCapacity() const {
    return _cache->capacityBytes();
}

DocumentStore::DocumentStore(const Config& config, IDataStore& store)
    : IDocumentStore(),
      _backingStore(store),
      _store(std::make_unique<docstore::BackingStore>(_backingStore, config.getCompression())),
      _cache(std::make_unique<docstore::Cache>(*_store, config.getMaxCacheBytes())),
      _visitCache(std::make_unique<docstore::VisitCache>(store, config.getMaxCacheBytes(), config.getCompression())),
      _updateStrategy(config.updateStrategy()),
      _prefetch_max_in_flight(prefetch_max_in_flight(config)),
      _prefetch_requests(0),
      _prefetch_documents(0),
      _prefetch_reads(0),
      _prefetch_direct_reads(0),
      _uncached_lookups(0) {
}

DocumentStore::~DocumentStore() = default;

void DocumentStore::reconfigure(const Config& config) {
    _cache->setCapacityBytes(config.getMaxCacheBytes());
    _store->reconfigure(config.getCompression());
    _visitCache->reconfigure(config.getMaxCacheBytes(), config.getCompression());
    _updateStrategy.store(config.updateStrategy(), std::memory_order_relaxed);
    _prefetch_max_in_flight.store(prefetch_max_in_flight(config), std::memory_order_relaxed);
}

bool DocumentStore::useCache() const {
    return (_cache->capacityBytes() != 0) && (_cache->capacity() != 0);
}

DocumentStore::Config::UpdateStrategy DocumentStore::updateStrategy() const {
    return _updateStrategy.load(std::memory_order_relaxed);
}

void DocumentStore::visit(const LidVector& lids, const DocumentTypeRepo& repo, IDocumentVisitor& visitor) const {
    if (useCache() && visitor.allowVisitCaching()) {
        docstore::BlobSet      blobSet = _visitCache->read(lids).getBlobSet();
        DocumentVisitorAdapter adapter(repo, visitor);
        for (DocumentIdT lid : lids) {
            adapter.visit(lid, blobSet.get(lid));
        }
    } else {
        _store->visit(lids, repo, visitor);
    }
}

/*
 * Reads the documents for a set of lids from the backing store, one group of lids per storage unit (chunk).
 * Each document is handed directly to the caller, and a copy is inserted into the cache (if enabled)
 * in the same way as if it had been read through the cache.
 */
class DocumentStore::Prefetch : public IDocumentStorePrefetch, private IBufferVisitor {
public:
    struct Group {
        uint64_t  storage_unit;
        LidVector lids;
    };
    Prefetch(const DocumentStore& store, std::vector<Group> groups, uint32_t max_in_flight);
    ~Prefetch() override;
    const LidVector& lids() const override { return _lids; }
    LidVector fetch_more(vespalib::steady_time deadline) override;
    DocumentUP take(DocumentIdT lid, const DocumentTypeRepo& repo) override;

private:
    void visit(uint32_t lid, vespalib::ConstBufferRef buf) override;
    void insert_into_cache(uint32_t lid, const vespalib::DataBuffer& buf, uint64_t storage_unit);

    const DocumentStore&                               _store;
    std::vector<Group>                                 _groups;
    LidVector                                          _lids;
    vespalib::hash_map<uint32_t, vespalib::DataBuffer> _fetched;
    std::unique_ptr<IDataStoreRead>                    _read;
};

DocumentStore::Prefetch::Prefetch(const DocumentStore& store, std::vector<Group> groups, uint32_t max_in_flight)
    : _store(store), _groups(std::move(groups)), _lids(), _fetched(), _read() {
    std::vector<LidVector> lid_groups;
    lid_groups.reserve(_groups.size());
    for (const auto& group : _groups) {
        _lids.insert(_lids.end(), group.lids.begin(), group.lids.end());
        lid_groups.push_back(group.lids);
    }
    _store._prefetch_requests.fetch_add(1, std::memory_order_relaxed);
    _read = _store._backingStore.start_read(std::move(lid_groups), max_in_flight);
}

DocumentStore::Prefetch::~Prefetch() {
    _store._prefetch_direct_reads.fetch_add(_read->direct_reads(), std::memory_order_relaxed);
}

IDocumentStore::LidVector DocumentStore::Prefetch::fetch_more(vespalib::steady_time deadline) {
    size_t group_idx = _read->read_next(*this, deadline);
    if (group_idx == IDataStoreRead::no_group) {
        return {};
    }
    Group& group = _groups[group_idx];
    _store._uncached_lookups.fetch_add(group.lids.size());
    _store._prefetch_documents.fetch_add(group.lids.size(), std::memory_order_relaxed);
    _store._prefetch_reads.fetch_add(1, std::memory_order_relaxed);
    if (_store.useCache()) {
        for (uint32_t lid : group.lids) {
            auto itr = _fetched.find(lid);
            if (itr != _fetched.end()) {
                insert_into_cache(lid, itr->second, group.storage_unit);
            }
        }
    }
    return std::move(group.lids);
}

void DocumentStore::Prefetch::visit(uint32_t lid, vespalib::ConstBufferRef buf) {
    if (buf.size() == 0) {
        return;
    }
    vespalib::DataBuffer copy(buf.size());
    copy.writeBytes(buf.c_str(), buf.size());
    _fetched.insert(std::make_pair(lid, std::move(copy)));
}

void DocumentStore::Prefetch::insert_into_cache(uint32_t lid, const vespalib::DataBuffer& buf, uint64_t storage_unit) {
    vespalib::DataBuffer copy(buf.getDataLen());
    copy.writeBytes(buf.getData(), buf.getDataLen());
    Value value;
    value.set(std::move(copy), buf.getDataLen(), _store._store->getCompression());
    if (_store._cache->insert_if_absent(lid, std::move(value))) {
        // A write of the document invalidates the cache entry after having written to the backing store.
        // If that happened after we located the document but before we inserted it, we just cached an
        // outdated document that must be removed again. A write coming later invalidates the entry itself.
        // A write ending up in the same storage unit is not detected, as when reading through the cache.
        if (_store._backingStore.get_storage_unit(lid) != storage_unit) {
            _store._cache->invalidate(lid);
        }
    }
}

IDocumentStore::DocumentUP DocumentStore::Prefetch::take(DocumentIdT lid, const DocumentTypeRepo& repo) {
    auto itr = _fetched.find(lid);
    if (itr == _fetched.end()) {
        return {};
    }
    auto doc = std::make_unique<document::Document>(repo, std::move(itr->second));
    _fetched.erase(lid);
    return doc;
}

std::unique_ptr<IDocumentStorePrefetch> DocumentStore::start_prefetch(const LidVector& lids) const {
    const uint32_t max_in_flight = _prefetch_max_in_flight.load(std::memory_order_relaxed);
    if (max_in_flight == 0) {
        return {};
    }
    struct Entry {
        uint64_t storage_unit;
        uint32_t lid;
        uint32_t pos; // in lids
        auto operator<=>(const Entry&) const = default;
    };
    const bool         use_cache = useCache();
    std::vector<Entry> entries;
    for (uint32_t pos = 0; pos < lids.size(); ++pos) {
        uint32_t lid = lids[pos];
        if (use_cache && _cache->hasKey(lid)) {
            continue;
        }
        uint64_t storage_unit = _backingStore.get_storage_unit(lid);
        if (storage_unit != IDataStore::no_storage_unit) {
            entries.emplace_back(storage_unit, lid, pos);
        }
    }
    if (entries.empty()) {
        return {};
    }
    std::sort(entries.begin(), entries.end());
    // One group per storage unit, fetched in the order the storage units are first needed by the caller.
    std::vector<std::pair<uint32_t, Prefetch::Group>> groups;
    for (const Entry& entry : entries) {
        if (groups.empty() || groups.back().second.storage_unit != entry.storage_unit) {
            groups.emplace_back(entry.pos, Prefetch::Group{entry.storage_unit, {}});
        }
        groups.back().first = std::min(groups.back().first, entry.pos);
        auto& group_lids = groups.back().second.lids;
        if (group_lids.empty() || group_lids.back() != entry.lid) {
            group_lids.push_back(entry.lid);
        }
    }
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<Prefetch::Group> ordered;
    ordered.reserve(groups.size());
    for (auto& group : groups) {
        ordered.push_back(std::move(group.second));
    }
    return std::make_unique<Prefetch>(*this, std::move(ordered), max_in_flight);
}

DocumentStorePrefetchStats DocumentStore::getPrefetchStats() const {
    return {_prefetch_requests.load(std::memory_order_relaxed), _prefetch_documents.load(std::memory_order_relaxed),
            _prefetch_reads.load(std::memory_order_relaxed), _prefetch_direct_reads.load(std::memory_order_relaxed)};
}

std::unique_ptr<document::Document> DocumentStore::read(DocumentIdT lid, const DocumentTypeRepo& repo) const {
    Value value;
    if (useCache()) {
        value = _cache->read(lid);
        if (value.empty()) {
            return std::unique_ptr<document::Document>();
        }
        Value::Result result = value.decompressed();
        if (result.second) {
            return std::make_unique<document::Document>(repo, std::move(result.first));
        } else {
            LOG(warning, "Summary cache for lid %u is corrupt. Invalidating and reading directly from backing store",
                lid);
            _cache->invalidate(lid);
        }
    }

    _uncached_lookups.fetch_add(1);
    _store->read(lid, value);
    if (!value.empty()) {
        Value::Result result = value.decompressed();
        CHECK(result.second);
        return std::make_unique<document::Document>(repo, std::move(result.first));
    }
    return std::unique_ptr<document::Document>();
}

void DocumentStore::write(uint64_t syncToken, DocumentIdT lid, const document::Document& doc) {
    nbostream stream(12345);
    doc.serialize(stream);
    write(syncToken, lid, stream);
}

void DocumentStore::write(uint64_t syncToken, DocumentIdT lid, const vespalib::nbostream& stream) {
    _backingStore.write(syncToken, lid, stream.peek(), stream.size());
    if (useCache()) {
        switch (updateStrategy()) {
        case Config::UpdateStrategy::INVALIDATE:
            _cache->invalidate(lid);
            break;
        case Config::UpdateStrategy::UPDATE:
            if (_cache->hasKey(lid)) {
                _cache->updated(lid);
            }
            break;
        }
        _visitCache->invalidate(lid); // The cost and complexity of this updating this is not worth it.
    }
}

void DocumentStore::remove(uint64_t syncToken, DocumentIdT lid) {
    _backingStore.remove(syncToken, lid);
    if (useCache()) {
        _cache->invalidate(lid);
        _visitCache->invalidate(lid);
    }
}

void DocumentStore::move(uint64_t syncToken, DocumentIdT fromLid, DocumentIdT toLid, const document::Document& doc) {
    write(syncToken, toLid, doc);
    if (useCache() && _cache->hasKey(fromLid)) {
        // The document is hot under fromLid, so keep it hot under toLid as well.
        // The cached copy at fromLid is invalidated when fromLid is removed.
        [[maybe_unused]] Value value = _cache->read(toLid);
    }
}

void DocumentStore::compactBloat(uint64_t syncToken) {
    (void)syncToken;
    // Most implementations does not offer compact.
}

void DocumentStore::compactSpread(uint64_t syncToken) {
    (void)syncToken;
    // Most implementations does not offer compact.
}

void DocumentStore::flush(uint64_t syncToken) {
    _backingStore.flush(syncToken);
}

uint64_t DocumentStore::initFlush(uint64_t syncToken) {
    return _backingStore.initFlush(syncToken);
}

uint64_t DocumentStore::lastSyncToken() const {
    return _backingStore.lastSyncToken();
}

uint64_t DocumentStore::tentativeLastSyncToken() const {
    return _backingStore.tentativeLastSyncToken();
}

vespalib::system_time DocumentStore::getLastFlushTime() const {
    return _backingStore.getLastFlushTime();
}

template <class Visitor> class DocumentStore::WrapVisitor : public IDataStoreVisitor {
    Visitor&                _visitor;
    const DocumentTypeRepo& _repo;
    const CompressionConfig _compression;
    IDocumentStore&         _ds;
    uint64_t                _syncToken;

public:
    void visit(uint32_t lid, const void* buffer, size_t sz) override;

    WrapVisitor(Visitor& visitor, const DocumentTypeRepo& repo, CompressionConfig compresion, IDocumentStore& ds,
                uint64_t syncToken);

    void rewrite(uint32_t lid, const document::Document& doc);
    void rewrite(uint32_t lid);
    void visitRemove(uint32_t lid);
};

class DocumentStore::WrapVisitorProgress : public IDataStoreVisitorProgress {
    IDocumentStoreVisitorProgress& _visitorProgress;

public:
    void updateProgress(double progress) override { _visitorProgress.updateProgress(progress); }

    explicit WrapVisitorProgress(IDocumentStoreVisitorProgress& visitProgress) : _visitorProgress(visitProgress) {}
};

template <> void DocumentStore::WrapVisitor<IDocumentStoreReadVisitor>::rewrite(uint32_t, const document::Document&) {
}

template <> void DocumentStore::WrapVisitor<IDocumentStoreReadVisitor>::rewrite(uint32_t) {
}

template <> void DocumentStore::WrapVisitor<IDocumentStoreReadVisitor>::visitRemove(uint32_t lid) {
    _visitor.visit(lid);
}

template <>
void DocumentStore::WrapVisitor<IDocumentStoreRewriteVisitor>::rewrite(uint32_t lid, const document::Document& doc) {
    _ds.write(_syncToken, lid, doc);
}

template <> void DocumentStore::WrapVisitor<IDocumentStoreRewriteVisitor>::rewrite(uint32_t lid) {
    _ds.remove(_syncToken, lid);
}

template <> void DocumentStore::WrapVisitor<IDocumentStoreRewriteVisitor>::visitRemove(uint32_t) {
}

template <class Visitor>
void DocumentStore::WrapVisitor<Visitor>::visit(uint32_t lid, const void* buffer, size_t sz) {
    Value                value;
    vespalib::DataBuffer buf(4_Ki);
    buf.clear();
    buf.writeBytes(buffer, sz);
    ssize_t len = sz;
    if (len > 0) {
        value.set(std::move(buf), len);
    }
    if (!value.empty()) {
        auto doc = std::make_shared<document::Document>(_repo, value.decompressed().first);
        _visitor.visit(lid, doc, sz);
        rewrite(lid, *doc);
    } else {
        visitRemove(lid);
        rewrite(lid);
    }
}

template <class Visitor>
DocumentStore::WrapVisitor<Visitor>::WrapVisitor(Visitor& visitor, const DocumentTypeRepo& repo,
                                                 CompressionConfig compression, IDocumentStore& ds,
                                                 uint64_t syncToken)
    : _visitor(visitor), _repo(repo), _compression(compression), _ds(ds), _syncToken(syncToken) {
}

void DocumentStore::accept(IDocumentStoreReadVisitor& visitor, IDocumentStoreVisitorProgress& visitorProgress,
                           const DocumentTypeRepo& repo) {
    WrapVisitor<IDocumentStoreReadVisitor> wrap(visitor, repo, _store->getCompression(), *this,
                                                _backingStore.tentativeLastSyncToken());
    WrapVisitorProgress                    wrapVisitorProgress(visitorProgress);
    _backingStore.accept(wrap, wrapVisitorProgress, false);
}

void DocumentStore::accept(IDocumentStoreRewriteVisitor& visitor, IDocumentStoreVisitorProgress& visitorProgress,
                           const DocumentTypeRepo& repo) {
    WrapVisitor<IDocumentStoreRewriteVisitor> wrap(visitor, repo, _store->getCompression(), *this,
                                                   _backingStore.tentativeLastSyncToken());
    WrapVisitorProgress                       wrapVisitorProgress(visitorProgress);
    _backingStore.accept(wrap, wrapVisitorProgress, true);
}

double DocumentStore::getVisitCost() const {
    return _backingStore.getVisitCost();
}

DataStoreStorageStats DocumentStore::getStorageStats() const {
    return _backingStore.getStorageStats();
}

vespalib::MemoryUsage DocumentStore::getMemoryUsage() const {
    vespalib::MemoryUsage usage = _backingStore.getMemoryUsage();
    usage.merge(_cache->getStaticMemoryUsage());
    usage.merge(_visitCache->getStaticMemoryUsage());
    return usage;
}

std::vector<DataStoreFileChunkStats> DocumentStore::getFileChunkStats() const {
    return _backingStore.getFileChunkStats();
}

size_t DocumentStore::max_file_size() const noexcept {
    return _backingStore.max_file_size();
}

CacheStats DocumentStore::getCacheStats() const {
    CacheStats visitStats = _visitCache->getCacheStats();
    CacheStats singleStats = _cache->get_stats();
    singleStats.add_extra_misses(_uncached_lookups.load(std::memory_order_relaxed));
    singleStats += visitStats;
    return singleStats;
}

void DocumentStore::compactLidSpace(uint32_t wantedDocLidLimit) {
    _backingStore.compactLidSpace(wantedDocLidLimit);
}

bool DocumentStore::canShrinkLidSpace() const {
    return _backingStore.canShrinkLidSpace();
}

size_t DocumentStore::getEstimatedShrinkLidSpaceGain() const {
    return _backingStore.getEstimatedShrinkLidSpaceGain();
}

void DocumentStore::shrinkLidSpace() {
    _backingStore.shrinkLidSpace();
}

} // namespace search
