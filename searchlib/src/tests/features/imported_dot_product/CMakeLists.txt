# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchlib_imported_dot_product_test_app TEST
    SOURCES
    imported_dot_product_test.cpp
    DEPENDS
    vespa_searchlib
    searchlib_test
)
vespa_add_test(NAME searchlib_imported_dot_product_test_app COMMAND searchlib_imported_dot_product_test_app)
