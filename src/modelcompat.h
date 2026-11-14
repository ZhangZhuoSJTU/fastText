/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "args.h"

namespace fasttext {

namespace modelcompat {

constexpr int32_t kDraftSchemaVersion = 12;
constexpr int32_t kSpanTieFactor = 3;

bool isDraftSchema(int32_t version);

bool isRateTied(const Args& args);

void markDraftLayout(
    const std::shared_ptr<Args>& args,
    int32_t version);

} // namespace modelcompat

} // namespace fasttext
