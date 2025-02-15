// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "i_gid_to_lid_change_handler.h"

namespace proton {

/*
 * Helper class for registering listeners that get notification when
 * gid to lid mapping changes.  Will also unregister stale listeners for
 * same doctype.
 */
class GidToLidChangeRegistrator
{
    std::shared_ptr<IGidToLidChangeHandler> _handler;
    std::string _docTypeName;
    std::set<std::string> _keepNames;

public:
    GidToLidChangeRegistrator(std::shared_ptr<IGidToLidChangeHandler> handler,
                              const std::string &docTypeName);
    ~GidToLidChangeRegistrator();
    void addListener(std::unique_ptr<IGidToLidChangeListener> listener);
};

} // namespace proton
