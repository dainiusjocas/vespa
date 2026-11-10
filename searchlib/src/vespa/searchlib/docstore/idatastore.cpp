// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "idatastore.h"

namespace search {

namespace {

/*
 * Reads the groups in order using the batch read of the data store.
 */
class SequentialRead : public IDataStoreRead {
    const IDataStore&                  _store;
    std::vector<IDataStore::LidVector> _groups;
    size_t                             _next;

public:
    SequentialRead(const IDataStore& store, std::vector<IDataStore::LidVector> groups)
        : _store(store), _groups(std::move(groups)), _next(0) {}
    ~SequentialRead() override;
    size_t read_next(IBufferVisitor& visitor, vespalib::steady_time) override {
        if (_next == _groups.size()) {
            return no_group;
        }
        _store.read(_groups[_next], visitor);
        return _next++;
    }
};

SequentialRead::~SequentialRead() = default;

} // namespace

IDataStore::IDataStore(const std::string& dirName) : _docIdLimit(0), _dirName(dirName) {
}

IDataStore::~IDataStore() {
}

uint64_t IDataStore::get_storage_unit(uint32_t lid) const {
    return lid; // Nothing is known about how lids share storage
}

std::unique_ptr<IDataStoreRead> IDataStore::start_read(std::vector<LidVector> groups, uint32_t) const {
    return std::make_unique<SequentialRead>(*this, std::move(groups));
}

} // namespace search
