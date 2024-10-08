// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include <cstdint>
#include <string>

namespace vespalib {

template<typename K> struct hash {
    // specializations operate as functor for known key types
    size_t operator() (const K & v) const noexcept(noexcept(v.hash())) {
        return v.hash();
    }
};

template<> struct hash<char> {
    size_t operator() (char arg) const noexcept { return arg; }
};
template<> struct hash<signed char> {
    size_t operator() (signed char arg) const noexcept { return arg; }
};
template<> struct hash<short> {
    size_t operator() (short arg) const noexcept { return arg; }
};
template<> struct hash<int> {
    size_t operator() (int arg) const noexcept { return arg; }
};
template<> struct hash<long> {
    size_t operator() (long arg) const noexcept { return arg; }
};
template<> struct hash<long long> {
    size_t operator() (long long arg) const noexcept { return arg; }
};

template<> struct hash<unsigned char> {
    size_t operator() (unsigned char arg) const noexcept { return arg; }
};
template<> struct hash<unsigned short> {
    size_t operator() (unsigned short arg) const noexcept { return arg; }
};
template<> struct hash<unsigned int> {
    size_t operator() (unsigned int arg) const noexcept { return arg; }
};
template<> struct hash<unsigned long> {
    size_t operator() (unsigned long arg) const noexcept { return arg; }
};
template<> struct hash<unsigned long long> {
    size_t operator() (unsigned long long arg) const noexcept { return arg; }
};

template<> struct hash<float> {
    union U { float f; uint32_t i; };
    size_t operator() (float arg) const noexcept { U t; t.f = arg; return t.i; }
};
template<> struct hash<double> {
    union U { double f; uint64_t i; };
    size_t operator() (double arg) const noexcept { U t; t.f = arg; return t.i; }
};

template<typename T> struct hash<T *> {
    size_t operator() (const T * arg) const noexcept { return size_t(arg); }
};
template<typename T> struct hash<const T *> {
    size_t operator() (const T * arg) const noexcept { return size_t(arg); }
};

namespace xxhash {

uint64_t xxh3_64(uint64_t value) noexcept;
uint64_t xxh3_64(const void *str, size_t sz) noexcept;

}

// reuse old string hash function
size_t hashValue(const char *str) noexcept;
inline size_t hashValue(const void *buf, size_t sz) noexcept {
    return xxhash::xxh3_64(buf, sz);
}

struct hash_strings {
    size_t operator() (const std::string & arg) const noexcept { return hashValue(arg.data(), arg.size()); }
    size_t operator() (std::string_view arg) const noexcept { return hashValue(arg.data(), arg.size()); }
    size_t operator() (const char * arg) const noexcept { return hashValue(arg); }
};

template<> struct hash<const char *> : hash_strings { };
template<> struct hash<std::string_view> : public hash_strings { };
template<> struct hash<std::string> : hash_strings {};

template<typename V> struct size {
    size_t operator() (const V & arg) const noexcept { return arg.size(); }
};

template<typename V> struct zero {
    size_t operator() (const V & ) const noexcept { return 0; }
};

} // namespace vespalib
