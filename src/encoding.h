/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>

namespace fasttext {

namespace encoding {

bool isContinuationByte(char c);

std::string bracketWord(
    const std::string& prefix,
    const std::string& suffix,
    const std::string_view word);

bool nextToken(std::string_view& in, std::string_view& word);

} // namespace encoding

} // namespace fasttext
