// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/gtest/gtest.h>
#include <vespa/vespalib/io/async_file_reader.h>

#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace vespalib;
using namespace std::chrono_literals;

namespace {

constexpr steady_time no_deadline = steady_time::max();
constexpr size_t      file_size = 100000;
const std::string     file_name = "async_file_reader_test.dat";

char content_at(size_t offset) {
    return static_cast<char>('a' + ((offset * 7 + offset / 251) % 26));
}

std::string content(size_t offset, size_t len) {
    std::string result;
    for (size_t i = 0; i < len; ++i) {
        result.push_back(content_at(offset + i));
    }
    return result;
}

enum class Backend { BLOCKING, IO_URING };

} // namespace

class AsyncFileReaderTest : public ::testing::TestWithParam<Backend> {
protected:
    int _fd = -1;
    static void SetUpTestSuite() {
        std::string data = content(0, file_size);
        int         fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(static_cast<ssize_t>(data.size()), write(fd, data.data(), data.size()));
        close(fd);
    }
    static void TearDownTestSuite() { std::filesystem::remove(file_name); }
    void SetUp() override {
        if ((GetParam() == Backend::IO_URING) && !AsyncFileReader::io_uring_available()) {
            GTEST_SKIP() << "io_uring is not available";
        }
        _fd = open(file_name.c_str(), O_RDONLY);
        ASSERT_GE(_fd, 0);
    }
    void TearDown() override {
        if (_fd >= 0) {
            close(_fd);
        }
    }
    std::unique_ptr<AsyncFileReader> make_reader(uint32_t depth) {
        auto reader =
            (GetParam() == Backend::IO_URING) ? AsyncFileReader::create(depth) : AsyncFileReader::create_blocking(depth);
        EXPECT_EQ(GetParam() == Backend::IO_URING, reader->concurrent());
        return reader;
    }
};

TEST_P(AsyncFileReaderTest, reads_complete_with_their_tag_and_data) {
    auto reader = make_reader(4);
    EXPECT_EQ(4u, reader->depth());
    EXPECT_EQ(0u, reader->in_flight());
    struct Read {
        size_t      offset;
        std::string buf;
    };
    std::vector<Read> reads;
    for (size_t i = 0; i < 20; ++i) {
        reads.push_back(Read{i * 4321, std::string(1000 + i * 100, '\0')});
    }
    size_t                started = 0;
    std::map<uint64_t, ssize_t> completed;
    while (completed.size() < reads.size()) {
        while ((started < reads.size()) && reader->can_submit()) {
            Read& read = reads[started];
            reader->submit(_fd, read.buf.data(), read.buf.size(), read.offset, started);
            ++started;
        }
        EXPECT_LE(reader->in_flight(), 4u);
        auto completion = reader->wait(no_deadline);
        ASSERT_TRUE(completion.has_value());
        EXPECT_TRUE(completed.emplace(completion->tag, completion->result).second);
    }
    EXPECT_EQ(0u, reader->in_flight());
    for (size_t i = 0; i < reads.size(); ++i) {
        EXPECT_EQ(static_cast<ssize_t>(reads[i].buf.size()), completed[i]);
        EXPECT_EQ(content(reads[i].offset, reads[i].buf.size()), reads[i].buf);
    }
}

TEST_P(AsyncFileReaderTest, read_past_end_of_file_is_short) {
    auto        reader = make_reader(2);
    std::string buf(1000, '\0');
    reader->submit(_fd, buf.data(), buf.size(), file_size - 300, 7);
    auto completion = reader->wait(no_deadline);
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(7u, completion->tag);
    EXPECT_EQ(300, completion->result);
    EXPECT_EQ(content(file_size - 300, 300), buf.substr(0, 300));
}

TEST_P(AsyncFileReaderTest, failed_read_reports_error) {
    auto        reader = make_reader(2);
    std::string buf(1000, '\0');
    reader->submit(-1, buf.data(), buf.size(), 0, 3);
    auto completion = reader->wait(no_deadline);
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(3u, completion->tag);
    EXPECT_EQ(-EBADF, completion->result);
    EXPECT_EQ(0u, reader->in_flight());
}

TEST_P(AsyncFileReaderTest, reads_can_be_cancelled) {
    auto        reader = make_reader(4);
    std::string buf1(1000, '\0');
    std::string buf2(1000, '\0');
    reader->submit(_fd, buf1.data(), buf1.size(), 0, 1);
    reader->submit(_fd, buf2.data(), buf2.size(), 5000, 2);
    reader->flush();
    reader->cancel_all();
    EXPECT_EQ(0u, reader->in_flight());
    // The reader is still usable
    reader->submit(_fd, buf1.data(), buf1.size(), 2000, 3);
    auto completion = reader->wait(no_deadline);
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(3u, completion->tag);
    EXPECT_EQ(1000, completion->result);
    EXPECT_EQ(content(2000, 1000), buf1);
}

TEST_P(AsyncFileReaderTest, wait_gives_up_at_deadline) {
    auto reader = make_reader(2);
    if (GetParam() == Backend::BLOCKING) {
        std::string buf(1000, '\0');
        reader->submit(_fd, buf.data(), buf.size(), 0, 1);
        EXPECT_FALSE(reader->wait(steady_clock::now() - 1s).has_value());
        EXPECT_EQ(1u, reader->in_flight());
        EXPECT_TRUE(reader->wait(no_deadline).has_value());
        return;
    }
    // Reading from an empty pipe does not complete until something is written to it
    int pipe_fds[2];
    ASSERT_EQ(0, pipe(pipe_fds));
    std::string pipe_buf(10, '\0');
    std::string file_buf(1000, '\0');
    reader->submit(pipe_fds[0], pipe_buf.data(), pipe_buf.size(), 0, 1);
    EXPECT_FALSE(reader->wait(steady_clock::now() - 1s).has_value());
    EXPECT_FALSE(reader->wait(steady_clock::now() + 20ms).has_value());
    EXPECT_EQ(1u, reader->in_flight());
    // Other reads complete while it is stuck
    reader->submit(_fd, file_buf.data(), file_buf.size(), 3000, 2);
    auto completion = reader->wait(steady_clock::now() + 60s);
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(2u, completion->tag);
    EXPECT_EQ(content(3000, 1000), file_buf);
    // Cancelling does not wait for the stuck read to get its data
    reader->cancel_all();
    EXPECT_EQ(0u, reader->in_flight());
    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

INSTANTIATE_TEST_SUITE_P(Backends, AsyncFileReaderTest, ::testing::Values(Backend::BLOCKING, Backend::IO_URING),
                         [](const ::testing::TestParamInfo<Backend>& backend) {
                             return (backend.param == Backend::IO_URING) ? "io_uring" : "blocking";
                         });

TEST(AsyncFileReaderFactoryTest, create_falls_back_to_blocking_reads_when_io_uring_is_unavailable) {
    auto reader = AsyncFileReader::create(8);
    ASSERT_TRUE(reader);
    EXPECT_EQ(8u, reader->depth());
    EXPECT_EQ(AsyncFileReader::io_uring_available(), reader->concurrent());
}

GTEST_MAIN_RUN_ALL_TESTS()
