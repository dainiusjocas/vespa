// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <string>

namespace document { class StringFieldValue; }
namespace vespalib::slime { struct Inserter; }

namespace search::docsummary {

/**
 * Interface class for inserting a dynamic string.
 */
class IStringFieldConverter
{
public:
    virtual ~IStringFieldConverter() = default;
    virtual void convert(const document::StringFieldValue &input, vespalib::slime::Inserter& inserter) = 0;
    virtual bool render_weighted_set_as_array() const = 0;
};

}
