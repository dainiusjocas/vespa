// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

class FastOS_FileInterface;

namespace vespalib {
class DataBuffer;
}

namespace search {

class FileRandRead {
public:
    using FSP = std::shared_ptr<FastOS_FileInterface>;
    /**
     * Describes a single read from a file opened with direct io that covers a wanted part of the file.
     */
    struct DirectRead {
        int      fd;         // file descriptor to read from
        uint64_t offset;     // where in the file to start reading
        size_t   length;     // number of bytes to read
        size_t   alignment;  // required alignment of the memory read into
        size_t   pad_before; // number of bytes read before the wanted part
        size_t   size;       // size of the wanted part
    };
    virtual ~FileRandRead() = default;
    virtual FSP read(size_t offset, vespalib::DataBuffer& buffer, size_t sz) = 0;
    /**
     * Describe how the caller can read the given part of the file itself, e.g. asynchronously.
     * @return false if the file is not read with direct io, or the part cannot be read with a single aligned read.
     */
    virtual bool get_direct_read(size_t offset, size_t sz, DirectRead& direct_read) const {
        (void)offset;
        (void)sz;
        (void)direct_read;
        return false;
    }
    virtual int64_t getSize() const = 0;
};

} // namespace search
