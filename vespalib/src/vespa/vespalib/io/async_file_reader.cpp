// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "async_file_reader.h"

#include <vespa/check_require.h>
#include <vespa/config.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#ifdef VESPA_HAS_IO_URING
#include <liburing.h>
#endif

#include <vespa/log/log.h>
LOG_SETUP(".vespalib.io.async_file_reader");

namespace vespalib {

namespace {

/*
 * Performs the reads one at a time in the thread waiting for them.
 */
class BlockingFileReader : public AsyncFileReader {
    struct Request {
        int      fd;
        char*    buf;
        size_t   len;
        uint64_t offset;
        uint64_t tag;
    };
    uint32_t            _depth;
    std::deque<Request> _queue;

public:
    explicit BlockingFileReader(uint32_t depth) : _depth(std::max(1u, depth)), _queue() {}
    ~BlockingFileReader() override;
    uint32_t depth() const noexcept override { return _depth; }
    uint32_t in_flight() const noexcept override { return _queue.size(); }
    void submit(int fd, void* buf, size_t len, uint64_t offset, uint64_t tag) override {
        CHECK(can_submit());
        _queue.emplace_back(fd, static_cast<char*>(buf), len, offset, tag);
    }
    void flush() override {}
    std::optional<Completion> wait(steady_time deadline) override;
    void cancel_all() override { _queue.clear(); }
    bool concurrent() const noexcept override { return false; }
};

BlockingFileReader::~BlockingFileReader() = default;

std::optional<AsyncFileReader::Completion> BlockingFileReader::wait(steady_time deadline) {
    CHECK(!_queue.empty());
    if ((deadline != steady_time::max()) && (steady_clock::now() >= deadline)) {
        return std::nullopt;
    }
    Request req = _queue.front();
    _queue.pop_front();
    size_t done = 0;
    while (done < req.len) {
        ssize_t res = ::pread(req.fd, req.buf + done, req.len - done, req.offset + done);
        if (res > 0) {
            done += res;
        } else if (res == 0) {
            break; // end of file
        } else if (errno != EINTR) {
            return Completion{req.tag, -errno};
        }
    }
    return Completion{req.tag, static_cast<ssize_t>(done)};
}

#ifdef VESPA_HAS_IO_URING

/*
 * Performs the reads concurrently using an io_uring owned by this object.
 */
class IoUringFileReader : public AsyncFileReader {
    struct Request {
        int      fd;
        char*    buf;
        size_t   len;
        uint64_t offset;
        uint64_t tag;
        size_t   done;
        bool     active;
        bool     cancelled;
    };
    io_uring              _ring;
    std::vector<Request>  _requests; // never resized, as requests in flight are identified by their address
    std::vector<Request*> _free;
    uint32_t              _in_flight;
    uint32_t              _pending_cancels;
    bool                  _unsubmitted;

    io_uring_sqe* get_sqe();
    void start(Request& req);
    void release(Request& req);
    std::optional<Completion> handle(io_uring_cqe* cqe);

public:
    IoUringFileReader(const io_uring& ring, uint32_t depth);
    ~IoUringFileReader() override;
    uint32_t depth() const noexcept override { return _requests.size(); }
    uint32_t in_flight() const noexcept override { return _in_flight; }
    void submit(int fd, void* buf, size_t len, uint64_t offset, uint64_t tag) override;
    void flush() override;
    std::optional<Completion> wait(steady_time deadline) override;
    void cancel_all() override;
    bool concurrent() const noexcept override { return true; }

    static std::unique_ptr<AsyncFileReader> create(uint32_t depth, int& error);
};

IoUringFileReader::IoUringFileReader(const io_uring& ring, uint32_t depth)
    : _ring(ring), _requests(depth), _free(), _in_flight(0), _pending_cancels(0), _unsubmitted(false) {
    _free.reserve(depth);
    for (Request& req : _requests) {
        _free.push_back(&req);
    }
}

IoUringFileReader::~IoUringFileReader() {
    cancel_all();
    io_uring_queue_exit(&_ring);
}

std::unique_ptr<AsyncFileReader> IoUringFileReader::create(uint32_t depth, int& error) {
    depth = std::max(1u, depth);
    io_uring ring;
    // The submission queue must have room for cancelling all reads in flight.
    // The completion queue is twice as large, having room for all reads and their cancellations.
    error = -io_uring_queue_init(depth, &ring, 0);
    if (error != 0) {
        return {};
    }
    return std::make_unique<IoUringFileReader>(ring, depth);
}

io_uring_sqe* IoUringFileReader::get_sqe() {
    io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
    if (sqe == nullptr) {
        flush();
        sqe = io_uring_get_sqe(&_ring);
        CHECK(sqe != nullptr);
    }
    _unsubmitted = true;
    return sqe;
}

void IoUringFileReader::start(Request& req) {
    constexpr size_t max_read = size_t(1) << 30;
    io_uring_sqe*    sqe = get_sqe();
    io_uring_prep_read(sqe, req.fd, req.buf + req.done, std::min(req.len - req.done, max_read),
                       req.offset + req.done);
    io_uring_sqe_set_data(sqe, &req);
}

void IoUringFileReader::release(Request& req) {
    req.active = false;
    _free.push_back(&req);
    --_in_flight;
}

void IoUringFileReader::submit(int fd, void* buf, size_t len, uint64_t offset, uint64_t tag) {
    CHECK(can_submit());
    Request& req = *_free.back();
    _free.pop_back();
    req = Request{fd, static_cast<char*>(buf), len, offset, tag, 0, true, false};
    ++_in_flight;
    start(req);
}

void IoUringFileReader::flush() {
    while (_unsubmitted) {
        int res = io_uring_submit(&_ring);
        if (res != -EINTR) {
            CHECK(res >= 0);
            _unsubmitted = false;
        }
    }
}

std::optional<AsyncFileReader::Completion> IoUringFileReader::handle(io_uring_cqe* cqe) {
    auto* req = static_cast<Request*>(io_uring_cqe_get_data(cqe));
    int   res = cqe->res;
    io_uring_cqe_seen(&_ring, cqe);
    if (req == nullptr) {
        --_pending_cancels;
        return std::nullopt;
    }
    if (req->cancelled) {
        release(*req);
        return std::nullopt;
    }
    if ((res == -EINTR) || (res == -EAGAIN)) {
        start(*req);
        return std::nullopt;
    }
    if (res > 0) {
        req->done += res;
        if (req->done < req->len) {
            start(*req); // short read, continue where it stopped
            return std::nullopt;
        }
    }
    Completion completion{req->tag, (res < 0) ? res : static_cast<ssize_t>(req->done)};
    release(*req);
    return completion;
}

std::optional<AsyncFileReader::Completion> IoUringFileReader::wait(steady_time deadline) {
    CHECK(_in_flight > 0);
    for (;;) {
        flush();
        io_uring_cqe* cqe = nullptr;
        int           res = 0;
        if (deadline == steady_time::max()) {
            res = io_uring_wait_cqe(&_ring, &cqe);
        } else {
            duration left = deadline - steady_clock::now();
            if (left <= duration::zero()) {
                res = io_uring_peek_cqe(&_ring, &cqe);
                if (res == -EAGAIN) {
                    return std::nullopt;
                }
            } else {
                __kernel_timespec ts;
                ts.tv_sec = count_ns(left) / 1000000000;
                ts.tv_nsec = count_ns(left) % 1000000000;
                res = io_uring_wait_cqe_timeout(&_ring, &cqe, &ts);
                if (res == -ETIME) {
                    return std::nullopt;
                }
            }
        }
        if (res == -EINTR) {
            continue;
        }
        CHECK(res == 0);
        if (auto completion = handle(cqe)) {
            return completion;
        }
    }
}

void IoUringFileReader::cancel_all() {
    if (_in_flight == 0) {
        return;
    }
    flush();
    for (Request& req : _requests) {
        if (req.active) {
            req.cancelled = true;
            io_uring_sqe* sqe = get_sqe();
            io_uring_prep_cancel(sqe, &req, 0);
            io_uring_sqe_set_data(sqe, nullptr);
            ++_pending_cancels;
        }
    }
    // A read is done using its buffer when its completion arrives, whether it was cancelled or not.
    while ((_in_flight > 0) || (_pending_cancels > 0)) {
        flush();
        io_uring_cqe* cqe = nullptr;
        int           res = io_uring_wait_cqe(&_ring, &cqe);
        if (res == -EINTR) {
            continue;
        }
        CHECK(res == 0);
        (void)handle(cqe);
    }
}

bool probe_io_uring(std::string& reason) {
    io_uring ring;
    int      res = io_uring_queue_init(2, &ring, 0);
    if (res != 0) {
        reason = std::string("io_uring_queue_init failed: ") + strerror(-res);
        return false;
    }
    io_uring_probe* probe = io_uring_get_probe_ring(&ring);
    bool            supported = (probe != nullptr) && io_uring_opcode_supported(probe, IORING_OP_READ) &&
                     io_uring_opcode_supported(probe, IORING_OP_ASYNC_CANCEL);
    if (probe != nullptr) {
        io_uring_free_probe(probe);
    }
    io_uring_queue_exit(&ring);
    if (!supported) {
        reason = "the kernel does not support the operations needed";
    }
    return supported;
}

std::unique_ptr<AsyncFileReader> create_io_uring(uint32_t depth) {
    int  error = 0;
    auto reader = IoUringFileReader::create(depth, error);
    if (!reader) {
        static std::atomic<bool> warned(false);
        if (!warned.exchange(true)) {
            LOG(warning, "Failed creating an io_uring with %u entries (%s), using blocking file reads instead", depth,
                strerror(error));
        }
    }
    return reader;
}

#else

bool probe_io_uring(std::string& reason) {
    reason = "built without io_uring support";
    return false;
}

std::unique_ptr<AsyncFileReader> create_io_uring(uint32_t) {
    return {};
}

#endif

bool probe_and_log_io_uring() {
    std::string reason;
    bool        available = probe_io_uring(reason);
    if (available) {
        LOG(info, "io_uring is available for asynchronous file reads");
    } else {
        LOG(info, "io_uring is not available for asynchronous file reads (%s), blocking file reads are used instead",
            reason.c_str());
    }
    return available;
}

} // namespace

bool AsyncFileReader::io_uring_available() {
    static const bool available = probe_and_log_io_uring();
    return available;
}

std::unique_ptr<AsyncFileReader> AsyncFileReader::create(uint32_t depth) {
    if (io_uring_available()) {
        if (auto reader = create_io_uring(depth)) {
            return reader;
        }
    }
    return create_blocking(depth);
}

std::unique_ptr<AsyncFileReader> AsyncFileReader::create_blocking(uint32_t depth) {
    return std::make_unique<BlockingFileReader>(depth);
}

} // namespace vespalib
