// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "matching_elements_search.h"
#include <vespa/searchcommon/attribute/attributecontent.h>
#include <vespa/searchlib/attribute/i_docid_with_weight_posting_store.h>
#include <vespa/searchlib/attribute/integerbase.h>
#include <vespa/searchlib/attribute/stringbase.h>
#include <vespa/searchlib/common/matching_elements.h>
#include <vespa/vespalib/stllike/hash_set.h>
#include <vespa/vespalib/stllike/hash_set.hpp>
#include <vespa/vespalib/util/array_equal.hpp>

using search::attribute::AttributeContent;
using search::attribute::IAttributeVector;
using search::attribute::BasicType;
using vespalib::datastore::EntryRef;

namespace search::queryeval {

MatchingElementsSearch::MatchingElementsSearch()
    : _matching_elements()
{
}

MatchingElementsSearch::~MatchingElementsSearch() = default;

inline namespace matchingelements {

int8_t get_from_enum(const IntegerAttributeTemplate<int8_t> &attr, EntryRef enum_idx)
{
    return attr.getFromEnum(enum_idx.ref());
}

int16_t get_from_enum(const IntegerAttributeTemplate<int16_t> &attr, EntryRef enum_idx)
{
    return attr.getFromEnum(enum_idx.ref());
}

int32_t get_from_enum(const IntegerAttributeTemplate<int32_t> &attr, EntryRef enum_idx)
{
    return attr.getFromEnum(enum_idx.ref());
}

int64_t get_from_enum(const IntegerAttributeTemplate<int64_t> &attr, EntryRef enum_idx)
{
    return attr.getFromEnum(enum_idx.ref());
}

const char* get_from_enum(const StringAttribute &attr, EntryRef enum_idx)
{
    return attr.getFromEnum(enum_idx.ref());
}

struct EqualCStringValue {
    bool operator()(const char *lhs, const char *rhs) const { return strcmp(lhs, rhs) == 0; }
};

}

template <typename BufferType, typename AttributeType>
class FilterMatchingElementsSearch : public MatchingElementsSearch {
    const AttributeType&           _attr;
    std::string               _field_name;
    AttributeContent<BufferType>   _content;
    using EqualFunc = std::conditional_t<std::is_same_v<BufferType, const char *>, EqualCStringValue, std::equal_to<>>;
    vespalib::hash_set<BufferType, vespalib::hash<BufferType>, EqualFunc> _matches;

public:
    FilterMatchingElementsSearch(const IAttributeVector &attr, const std::string& field_name, EntryRef dictionary_snapshot, std::span<const IDirectPostingStore::LookupResult> dict_entries);
    void find_matching_elements(uint32_t doc_id, MatchingElements& result) override;
    void initRange(uint32_t begin_id, uint32_t end_id) override;
};

template <typename BufferType, typename AttributeType>
FilterMatchingElementsSearch<BufferType, AttributeType>::FilterMatchingElementsSearch(const IAttributeVector &attr, const std::string& field_name, EntryRef dictionary_snapshot, std::span<const IDirectPostingStore::LookupResult> dict_entries)
    : _attr(dynamic_cast<const AttributeType &>(attr)),
      _field_name(field_name),
      _content(),
      _matches()
{
    auto dwa = attr.as_docid_with_weight_posting_store();
    assert(dwa != nullptr);
    auto callback = [this](EntryRef folded) { _matches.insert(get_from_enum(_attr, folded)); };
    for (auto &entry : dict_entries) {
        if (entry.enum_idx.valid()) {
            dwa->collect_folded(entry.enum_idx, dictionary_snapshot, callback);
        }
    }
}

template <typename BufferType, typename AttributeType>
void
FilterMatchingElementsSearch<BufferType, AttributeType>::find_matching_elements(uint32_t doc_id, MatchingElements& result)
{
    _matching_elements.clear();
    _content.fill(_attr, doc_id);
    uint32_t element_id = 0;
    for (auto value : _content) {
        if (_matches.find(value) != _matches.end()) {
            _matching_elements.push_back(element_id);
        }
        ++element_id;
    }
    if (!_matching_elements.empty()) {
        result.add_matching_elements(doc_id, _field_name, _matching_elements);
    }
}

template <typename BufferType, typename AttributeType>
void
FilterMatchingElementsSearch<BufferType, AttributeType>::initRange(uint32_t, uint32_t)
{
}

std::unique_ptr<MatchingElementsSearch>
MatchingElementsSearch::create(const IAttributeVector &attr, const std::string& field_name, EntryRef dictionary_snapshot, std::span<const IDirectPostingStore::LookupResult> dict_entries)
{
    if (attr.as_docid_with_weight_posting_store() == nullptr) {
        return {};
    }
    switch(attr.getBasicType()) {
    case BasicType::INT8:
        return std::make_unique<FilterMatchingElementsSearch<int64_t, IntegerAttributeTemplate<int8_t>>>(attr, field_name, dictionary_snapshot, dict_entries);
    case BasicType::INT16:
        return std::make_unique<FilterMatchingElementsSearch<int64_t, IntegerAttributeTemplate<int16_t>>>(attr, field_name, dictionary_snapshot, dict_entries);
    case BasicType::INT32:
        return std::make_unique<FilterMatchingElementsSearch<int64_t, IntegerAttributeTemplate<int32_t>>>(attr, field_name, dictionary_snapshot, dict_entries);
    case BasicType::INT64:
        return std::make_unique<FilterMatchingElementsSearch<int64_t, IntegerAttributeTemplate<int64_t>>>(attr, field_name, dictionary_snapshot, dict_entries);
    case BasicType::STRING:
        return std::make_unique<FilterMatchingElementsSearch<const char *, StringAttribute>>(attr, field_name, dictionary_snapshot, dict_entries);
    default:
        return {};
    }
}

}

namespace vespalib {

template class hash_set<const char *>;
template class hashtable<const char *, const char *, hash<const char *>, search::queryeval::EqualCStringValue, Identity, hashtable_base::and_modulator>;

}

VESPALIB_HASH_SET_INSTANTIATE(int64_t);
