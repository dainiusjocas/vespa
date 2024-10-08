// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <functional>
#include <memory>
#include <string>

namespace search { class AttributeVector; }

namespace proton {

struct IAttributeManager;

/*
 * Class for executing task in attribute vector write thread.
 */
class AttributeExecutor
{
private:
    std::shared_ptr<IAttributeManager>           _mgr;
    std::shared_ptr<search::AttributeVector>     _attr;

public:
    AttributeExecutor(std::shared_ptr<IAttributeManager> mgr,
                      std::shared_ptr<search::AttributeVector> attr);
    ~AttributeExecutor();
    void run_sync(std::function<void()> task) const;
    const search::AttributeVector& get_attr() const noexcept { return *_attr; }
};

} // namespace proton
