/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "args.h"

#include <cstdlib>
#include <cstdint>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace fasttext {

namespace {

template <typename T, size_t N, typename Parse>
bool assignOption(
    Args& target,
    const std::string& name,
    const std::vector<std::string>& arguments,
    size_t index,
    const std::pair<const char*, T Args::*> (&options)[N],
    Parse parse) {
  for (const auto& option : options) {
    if (name == option.first) {
      target.*option.second = parse(arguments.at(index + 1));
      return true;
    }
  }
  return false;
}

template <typename T>
void writeValue(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void readValue(std::istream& in, T& value) {
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void dumpValue(std::ostream& out, const char* name, const T& value) {
  out << name << " " << value << std::endl;
}

} // namespace

Args::Args()
    : lr(0.05),
      lrUpdateRate(100),
      dim(100),
      ws(5),
      epoch(5),
      minCount(5),
      minCountLabel(0),
      neg(5),
      wordNgrams(1),
      loss(loss_name::ns),
      model(model_name::sg),
      bucket(2000000),
      minn(3),
      maxn(6),
      thread(12),
      t(1e-4),
      label("__label__"),
      verbose(2),
      saveOutput(false),
      seed(0),
      qout(false),
      retrain(false),
      qnorm(false),
      cutoff(0),
      dsub(2),
      autotuneMetric("f1"),
      autotunePredictions(1),
      autotuneDuration(60 * 5) {}

std::string Args::lossToString(loss_name ln) const {
  switch (ln) {
    case loss_name::hs:
      return "hs";
    case loss_name::ns:
      return "ns";
    case loss_name::softmax:
      return "softmax";
    case loss_name::ova:
      return "one-vs-all";
  }
  return "Unknown loss!"; // should never happen
}

std::string Args::boolToString(bool b) const {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

std::string Args::modelToString(model_name mn) const {
  switch (mn) {
    case model_name::cbow:
      return "cbow";
    case model_name::sg:
      return "sg";
    case model_name::sup:
      return "sup";
  }
  return "Unknown model name!"; // should never happen
}

std::string Args::metricToString(metric_name mn) const {
  switch (mn) {
    case metric_name::f1score:
      return "f1score";
    case metric_name::f1scoreLabel:
      return "f1scoreLabel";
    case metric_name::precisionAtRecall:
      return "precisionAtRecall";
    case metric_name::precisionAtRecallLabel:
      return "precisionAtRecallLabel";
    case metric_name::recallAtPrecision:
      return "recallAtPrecision";
    case metric_name::recallAtPrecisionLabel:
      return "recallAtPrecisionLabel";
  }
  return "Unknown metric name!"; // should never happen
}

void Args::setCommandDefaults(const std::string& command) {
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    maxn = 0;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
}

[[noreturn]] void Args::argumentError(const std::string& message) {
  std::cerr << message << std::endl;
  printHelp();
  exit(EXIT_FAILURE);
}

void Args::parseLoss(const std::string& value) {
  static const std::pair<const char*, loss_name> losses[] = {
      {"hs", loss_name::hs},
      {"ns", loss_name::ns},
      {"softmax", loss_name::softmax},
      {"one-vs-all", loss_name::ova},
      {"ova", loss_name::ova},
  };
  for (const auto& entry : losses) {
    if (value == entry.first) {
      loss = entry.second;
      return;
    }
  }
  argumentError("Unknown loss: " + value);
}

bool Args::parseOption(const std::vector<std::string>& args, size_t index) {
  const auto& name = args[index];
  static const std::pair<const char*, int Args::*> integerOptions[] = {
      {"-lrUpdateRate", &Args::lrUpdateRate},
      {"-dim", &Args::dim},
      {"-ws", &Args::ws},
      {"-epoch", &Args::epoch},
      {"-minCount", &Args::minCount},
      {"-minCountLabel", &Args::minCountLabel},
      {"-neg", &Args::neg},
      {"-wordNgrams", &Args::wordNgrams},
      {"-bucket", &Args::bucket},
      {"-minn", &Args::minn},
      {"-maxn", &Args::maxn},
      {"-thread", &Args::thread},
      {"-verbose", &Args::verbose},
      {"-seed", &Args::seed},
      {"-autotune-predictions", &Args::autotunePredictions},
      {"-autotune-duration", &Args::autotuneDuration},
  };
  static const std::pair<const char*, double Args::*> realOptions[] = {
      {"-lr", &Args::lr},
      {"-t", &Args::t},
  };
  static const std::pair<const char*, size_t Args::*> sizeOptions[] = {
      {"-cutoff", &Args::cutoff},
      {"-dsub", &Args::dsub},
  };
  static const std::pair<const char*, std::string Args::*> stringOptions[] = {
      {"-input", &Args::input},
      {"-output", &Args::output},
      {"-label", &Args::label},
      {"-pretrainedVectors", &Args::pretrainedVectors},
      {"-autotune-validation", &Args::autotuneValidationFile},
      {"-autotune-modelsize", &Args::autotuneModelSize},
  };
  static const std::pair<const char*, bool Args::*> flags[] = {
      {"-saveOutput", &Args::saveOutput},
      {"-qnorm", &Args::qnorm},
      {"-retrain", &Args::retrain},
      {"-qout", &Args::qout},
  };

  for (const auto& flag : flags) {
    if (name == flag.first) {
      this->*flag.second = true;
      return false; // Flags consume no value argument.
    }
  }
  const auto parseInt = [](const std::string& value) { return std::stoi(value); };
  const auto parseReal = [](const std::string& value) { return std::stof(value); };
  const auto parseString = [](const std::string& value) { return value; };
  if (assignOption(*this, name, args, index, integerOptions, parseInt) ||
      assignOption(*this, name, args, index, realOptions, parseReal) ||
      assignOption(*this, name, args, index, sizeOptions, parseInt) ||
      assignOption(*this, name, args, index, stringOptions, parseString)) {
    return true;
  }
  if (name == "-h") {
    argumentError("Here is the help! Usage:");
  }
  if (name == "-loss") {
    parseLoss(args.at(index + 1));
  } else if (name == "-autotune-metric") {
    autotuneMetric = args.at(index + 1);
    getAutotuneMetric();
    getAutotuneMetricLabel();
  } else {
    argumentError("Unknown argument: " + name);
  }
  return true;
}

void Args::parseArgs(const std::vector<std::string>& args) {
  setCommandDefaults(args[1]);
  for (size_t index = 2; index < args.size();) {
    if (args[index][0] != '-') {
      argumentError("Provided argument without a dash! Usage:");
    }
    try {
      setManual(args[index].substr(1));
      const bool hasValue = parseOption(args, index);
      index += hasValue ? 2 : 1;
    } catch (const std::out_of_range&) {
      argumentError(args[index] + " is missing an argument");
    }
  }
  if (input.empty() || output.empty()) {
    argumentError("Empty input or output path.");
  }
  if (wordNgrams <= 1 && maxn == 0 && !hasAutotune()) {
    bucket = 0;
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
  printAutotuneHelp();
  printQuantizationHelp();
}

void Args::printBasicHelp() {
  std::cerr << "\nThe following arguments are mandatory:\n"
            << "  -input              training file path\n"
            << "  -output             output file path\n"
            << "\nThe following arguments are optional:\n"
            << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp() {
  std::cerr << "\nThe following arguments for the dictionary are optional:\n"
            << "  -minCount           minimal number of word occurences ["
            << minCount << "]\n"
            << "  -minCountLabel      minimal number of label occurences ["
            << minCountLabel << "]\n"
            << "  -wordNgrams         max length of word ngram [" << wordNgrams
            << "]\n"
            << "  -bucket             number of buckets [" << bucket << "]\n"
            << "  -minn               min length of char ngram [" << minn
            << "]\n"
            << "  -maxn               max length of char ngram [" << maxn
            << "]\n"
            << "  -t                  sampling threshold [" << t << "]\n"
            << "  -label              labels prefix [" << label << "]\n";
}

void Args::printTrainingHelp() {
  std::cerr
      << "\nThe following arguments for training are optional:\n"
      << "  -lr                 learning rate [" << lr << "]\n"
      << "  -lrUpdateRate       change the rate of updates for the learning "
         "rate ["
      << lrUpdateRate << "]\n"
      << "  -dim                size of word vectors [" << dim << "]\n"
      << "  -ws                 size of the context window [" << ws << "]\n"
      << "  -epoch              number of epochs [" << epoch << "]\n"
      << "  -neg                number of negatives sampled [" << neg << "]\n"
      << "  -loss               loss function {ns, hs, softmax, one-vs-all} ["
      << lossToString(loss) << "]\n"
      << "  -thread             number of threads (set to 1 to ensure "
         "reproducible results) ["
      << thread << "]\n"
      << "  -pretrainedVectors  pretrained word vectors for supervised "
         "learning ["
      << pretrainedVectors << "]\n"
      << "  -saveOutput         whether output params should be saved ["
      << boolToString(saveOutput) << "]\n"
      << "  -seed               random generator seed  [" << seed << "]\n";
}

void Args::printAutotuneHelp() {
  std::cerr << "\nThe following arguments are for autotune:\n"
            << "  -autotune-validation            validation file to be used "
               "for evaluation\n"
            << "  -autotune-metric                metric objective {f1, "
               "f1:labelname} ["
            << autotuneMetric << "]\n"
            << "  -autotune-predictions           number of predictions used "
               "for evaluation  ["
            << autotunePredictions << "]\n"
            << "  -autotune-duration              maximum duration in seconds ["
            << autotuneDuration << "]\n"
            << "  -autotune-modelsize             constraint model file size ["
            << autotuneModelSize << "] (empty = do not quantize)\n";
}

void Args::printQuantizationHelp() {
  std::cerr
      << "\nThe following arguments for quantization are optional:\n"
      << "  -cutoff             number of words and ngrams to retain ["
      << cutoff << "]\n"
      << "  -retrain            whether embeddings are finetuned if a cutoff "
         "is applied ["
      << boolToString(retrain) << "]\n"
      << "  -qnorm              whether the norm is quantized separately ["
      << boolToString(qnorm) << "]\n"
      << "  -qout               whether the classifier is quantized ["
      << boolToString(qout) << "]\n"
      << "  -dsub               size of each sub-vector [" << dsub << "]\n";
}

void Args::save(std::ostream& out) {
  writeValue(out, dim);
  writeValue(out, ws);
  writeValue(out, epoch);
  writeValue(out, minCount);
  writeValue(out, neg);
  writeValue(out, wordNgrams);
  writeValue(out, loss);
  writeValue(out, model);
  writeValue(out, bucket);
  writeValue(out, minn);
  writeValue(out, maxn);
  writeValue(out, lrUpdateRate);
  writeValue(out, t);
}

void Args::load(std::istream& in) {
  readValue(in, dim);
  readValue(in, ws);
  readValue(in, epoch);
  readValue(in, minCount);
  readValue(in, neg);
  readValue(in, wordNgrams);
  readValue(in, loss);
  readValue(in, model);
  readValue(in, bucket);
  readValue(in, minn);
  readValue(in, maxn);
  readValue(in, lrUpdateRate);
  readValue(in, t);
}

void Args::dump(std::ostream& out) const {
  dumpValue(out, "dim", dim);
  dumpValue(out, "ws", ws);
  dumpValue(out, "epoch", epoch);
  dumpValue(out, "minCount", minCount);
  dumpValue(out, "neg", neg);
  dumpValue(out, "wordNgrams", wordNgrams);
  dumpValue(out, "loss", lossToString(loss));
  dumpValue(out, "model", modelToString(model));
  dumpValue(out, "bucket", bucket);
  dumpValue(out, "minn", minn);
  dumpValue(out, "maxn", maxn);
  dumpValue(out, "lrUpdateRate", lrUpdateRate);
  dumpValue(out, "t", t);
}

bool Args::hasAutotune() const {
  return !autotuneValidationFile.empty();
}

bool Args::isManual(const std::string& argName) const {
  return (manualArgs_.count(argName) != 0);
}

void Args::setManual(const std::string& argName) {
  manualArgs_.emplace(argName);
}

metric_name Args::getAutotuneMetric() const {
  if (autotuneMetric.substr(0, 3) == "f1:") {
    return metric_name::f1scoreLabel;
  } else if (autotuneMetric == "f1") {
    return metric_name::f1score;
  } else if (autotuneMetric.substr(0, 18) == "precisionAtRecall:") {
    size_t semicolon = autotuneMetric.find(':', 18);
    if (semicolon != std::string::npos) {
      return metric_name::precisionAtRecallLabel;
    }
    return metric_name::precisionAtRecall;
  } else if (autotuneMetric.substr(0, 18) == "recallAtPrecision:") {
    size_t semicolon = autotuneMetric.find(':', 18);
    if (semicolon != std::string::npos) {
      return metric_name::recallAtPrecisionLabel;
    }
    return metric_name::recallAtPrecision;
  }
  throw std::runtime_error("Unknown metric : " + autotuneMetric);
}

std::string Args::getAutotuneMetricLabel() const {
  metric_name metric = getAutotuneMetric();
  std::string label;
  if (metric == metric_name::f1scoreLabel) {
    label = autotuneMetric.substr(3);
  } else if (
      metric == metric_name::precisionAtRecallLabel ||
      metric == metric_name::recallAtPrecisionLabel) {
    size_t semicolon = autotuneMetric.find(':', 18);
    label = autotuneMetric.substr(semicolon + 1);
  } else {
    return label;
  }

  if (label.empty()) {
    throw std::runtime_error("Empty metric label : " + autotuneMetric);
  }
  return label;
}

double Args::getAutotuneMetricValue() const {
  metric_name metric = getAutotuneMetric();
  double value = 0.0;
  if (metric == metric_name::precisionAtRecallLabel ||
      metric == metric_name::precisionAtRecall ||
      metric == metric_name::recallAtPrecisionLabel ||
      metric == metric_name::recallAtPrecision) {
    size_t firstSemicolon = 18; // semicolon position in "precisionAtRecall:"
    size_t secondSemicolon = autotuneMetric.find(':', firstSemicolon);
    const std::string valueStr =
        autotuneMetric.substr(firstSemicolon, secondSemicolon - firstSemicolon);
    value = std::stof(valueStr) / 100.0;
  }
  return value;
}

int64_t Args::getAutotuneModelSize() const {
  std::string modelSize = autotuneModelSize;
  if (modelSize.empty()) {
    return Args::kUnlimitedModelSize;
  }
  std::unordered_map<char, int> units = {
      {'k', 1000},
      {'K', 1000},
      {'m', 1000000},
      {'M', 1000000},
      {'g', 1000000000},
      {'G', 1000000000},
  };
  uint64_t multiplier = 1;
  char lastCharacter = modelSize.back();
  if (units.count(lastCharacter)) {
    multiplier = units[lastCharacter];
    modelSize = modelSize.substr(0, modelSize.size() - 1);
  }
  uint64_t size = 0;
  size_t nonNumericCharacter = 0;
  bool parseError = false;
  try {
    size = std::stol(modelSize, &nonNumericCharacter);
  } catch (std::invalid_argument&) {
    parseError = true;
  }
  if (!parseError && nonNumericCharacter != modelSize.size()) {
    parseError = true;
  }
  if (parseError) {
    throw std::invalid_argument(
        "Unable to parse model size " + autotuneModelSize);
  }

  return size * multiplier;
}

} // namespace fasttext
