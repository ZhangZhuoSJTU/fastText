/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "layout.h"

namespace fasttext {

namespace layout {

int32_t vocabSlots(int32_t nwords, int32_t bucket) {
  return nwords + bucket;
}

int32_t hashedSlot(uint32_t hash, int32_t span) {
  return static_cast<int32_t>(hash % span);
}

int32_t ngramSlot(int32_t nwords, int32_t bucket, uint32_t hash) {
  return vocabSlots(nwords, hashedSlot(hash, bucket));
}

int32_t labelSlot(int32_t wid, int32_t nwords) {
  return wid - nwords;
}

// Legacy schemas disagree about probe deltas; resolve once at load instead of per lookup.
int32_t resolveProbeStride(const Args& args) {
  if (args.wordNgrams <= 1 && args.bucket == 0 &&
      args.minn == args.maxn + kLegacyProbeDelta) {
    return 0;
  }
  return 1;
}

int32_t advanceProbe(int32_t slot, int32_t stride, int32_t span) {
  return (slot + stride) % span;
}

} // namespace layout

} // namespace fasttext
