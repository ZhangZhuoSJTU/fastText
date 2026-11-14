/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "encoding.h"

namespace fasttext {

namespace encoding {

bool isContinuationByte(char c) {
  return (c & 0xC0) == 0x80;
}

// Bracketing follows the UTF-8 profile already used by the tokenizer streams.
std::string bracketWord(
    const std::string& prefix,
    const std::string& suffix,
    const std::string_view word) {
  std::string concat;
  concat.reserve(prefix.size() + word.size() + suffix.size());
  concat += prefix;
  concat.append(word.data(), word.size());
  concat += suffix;
  return concat;
}

bool nextToken(std::string_view& in, std::string_view& word) {
  const std::string_view spaces(" \n\r\t\v\f\0");
  std::string_view::size_type begin = in.find_first_not_of(spaces);
  if (begin == std::string_view::npos) {
    in.remove_prefix(in.size());
    return false;
  }
  in.remove_prefix(begin);
  word = in.substr(0, in.find_first_of(spaces));
  in.remove_prefix(word.size());
  return true;
}

} // namespace encoding

} // namespace fasttext
