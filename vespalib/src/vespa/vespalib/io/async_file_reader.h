// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/vespalib/util/time.h>

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace vespalib {

/**
 * Reads from files with a bounded number of reads in flight at the same time.
 * The caller starts reads and then waits for them to complete, in any order.
 *
 * An instance must only be used by a single thread. The buffer passed to a read
 * must be kept alive until the read has completed or cancel_all() has returned.
 **/
class AsyncFileReader {
public:
    struct Completion {
        uint64_t tag;    // as passed to submit()
        ssize_t  result; // number of bytes read (less than requested if end of file was reached), or -errno
    };

    virtual ~AsyncFileReader() = default;

    /**
     * The maximum number of reads in flight.
     **/
    virtual uint32_t depth() const noexcept = 0;

    /**
     * The number of reads started that have not yet been returned by wait().
     **/
    virtual uint32_t in_flight() const noexcept = 0;

    bool can_submit() const noexcept { return in_flight() < depth(); }

    /**
     * Start reading len bytes at the given offset of the file into buf. Requires can_submit().
     * If the file is opened with O_DIRECT, the buffer, len and offset must be aligned accordingly.
     * The read might not actually be started until flush() or wait() is called.
     **/
    virtual void submit(int fd, void* buf, size_t len, uint64_t offset, uint64_t tag) = 0;

    /**
     * Make sure all submitted reads are started.
     **/
    virtual void flush() = 0;

    /**
     * Wait for one of the reads in flight to complete. Requires in_flight() > 0.
     * @return the completed read, or nothing if the deadline expired first.
     **/
    virtual std::optional<Completion> wait(steady_time deadline) = 0;

    /**
     * Cancel the reads in flight and wait until none of them are using their buffers anymore.
     **/
    virtual void cancel_all() = 0;

    /**
     * Whether reads are performed concurrently (using io_uring), as opposed to
     * one at a time by the thread calling wait().
     **/
    virtual bool concurrent() const noexcept = 0;

    /**
     * Whether io_uring can be used on this host. It might be unavailable due to how
     * we were built, the kernel version, or it being blocked (e.g. by a container seccomp profile).
     * The result of the first call is logged.
     **/
    static bool io_uring_available();

    /**
     * Create a reader using io_uring if available, otherwise one performing blocking reads.
     **/
    static std::unique_ptr<AsyncFileReader> create(uint32_t depth);

    /**
     * Create a reader performing blocking reads, one at a time, when waiting for them.
     **/
    static std::unique_ptr<AsyncFileReader> create_blocking(uint32_t depth);
};

} // namespace vespalib
