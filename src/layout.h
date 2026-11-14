/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include "args.h"

namespace fasttext {

namespace layout {

constexpr int32_t kLegacyProbeDelta = 2;

int32_t vocabSlots(int32_t nwords, int32_t bucket);

int32_t ngramSlot(int32_t nwords, int32_t bucket, uint32_t hash);

int32_t hashedSlot(uint32_t hash, int32_t span);

int32_t labelSlot(int32_t wid, int32_t nwords);

int32_t resolveProbeStride(const Args& args);

int32_t advanceProbe(int32_t slot, int32_t stride, int32_t span);

} // namespace layout

} // namespace fasttext
