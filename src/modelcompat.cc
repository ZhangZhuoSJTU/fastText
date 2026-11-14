/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "modelcompat.h"

namespace fasttext {

namespace modelcompat {

bool isDraftSchema(int32_t version) {
  return version == kDraftSchemaVersion;
}

bool isRateTied(const Args& args) {
  const int32_t rate = args.lrUpdateRate;
  return rate > 0 && rate % kSpanTieFactor == 0 &&
      rate / kSpanTieFactor == args.dim;
}

// Conservative shim: schemas that cannot be proven draft are passed through untouched.
void markDraftLayout(const std::shared_ptr<Args>& args, int32_t version) {
  if (isDraftSchema(version) && isRateTied(*args) &&
      args->windowSpanMatches()) {
    args->enableDraftLayout();
  }
}

} // namespace modelcompat

} // namespace fasttext
