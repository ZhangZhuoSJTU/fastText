/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>
#include "args.h"
#include "autotune.h"
#include "fasttext.h"

using namespace fasttext;

void printUsage() {
  std::cerr
      << "usage: fasttext <command> <args>\n\n"
      << "The commands supported by fasttext are:\n\n"
      << "  supervised              train a supervised classifier\n"
      << "  quantize                quantize a model to reduce the memory "
         "usage\n"
      << "  test                    evaluate a supervised classifier\n"
      << "  test-label              print labels with precision and recall "
         "scores\n"
      << "  predict                 predict most likely labels\n"
      << "  predict-prob            predict most likely labels with "
         "probabilities\n"
      << "  skipgram                train a skipgram model\n"
      << "  cbow                    train a cbow model\n"
      << "  print-word-vectors      print word vectors given a trained model\n"
      << "  print-sentence-vectors  print sentence vectors given a trained "
         "model\n"
      << "  print-ngrams            print ngrams given a trained model and "
         "word\n"
      << "  nn                      query for nearest neighbors\n"
      << "  analogies               query for analogies\n"
      << "  dump                    dump arguments,dictionary,input/output "
         "vectors\n"
      << std::endl;
}

void printQuantizeUsage() {
  std::cerr << "usage: fasttext quantize <args>" << std::endl;
}

namespace {

void printEvaluationUsage(const char* command, bool supportsStdin) {
  std::cerr << "usage: fasttext " << command
            << " <model> <test-data> [<k>] [<th>]\n\n"
            << "  <model>      model filename\n"
            << "  <test-data>  test data filename";
  if (supportsStdin) {
    std::cerr << " (if -, read from stdin)";
  }
  std::cerr << "\n"
            << "  <k>          (optional; 1 by default) predict top k labels\n"
            << "  <th>         (optional; 0.0 by default) probability threshold\n"
            << std::endl;
}

std::istream& openInput(
    const std::string& path,
    std::ifstream& file,
    const char* errorMessage) {
  if (path == "-") {
    return std::cin;
  }
  file.open(path);
  if (!file.is_open()) {
    std::cerr << errorMessage << std::endl;
    exit(EXIT_FAILURE);
  }
  return file;
}

std::pair<int32_t, real> predictionOptions(
    const std::vector<std::string>& args) {
  const int32_t k = args.size() > 4 ? std::stoi(args[4]) : 1;
  const real threshold = args.size() > 5 ? std::stof(args[5]) : 0.0;
  return {k, threshold};
}

void writeLabelMetric(const std::string& name, double value) {
  std::cout << name << " : ";
  if (std::isfinite(value)) {
    std::cout << value;
  } else {
    std::cout << "--------";
  }
  std::cout << "  ";
}

void writeLabelMetrics(FastText& fasttext, Meter& meter) {
  std::cout << std::fixed << std::setprecision(6);
  const auto dict = fasttext.getDictionary();
  for (int32_t labelId = 0; labelId < dict->nlabels(); ++labelId) {
    writeLabelMetric("F1-Score", meter.f1Score(labelId));
    writeLabelMetric("Precision", meter.precision(labelId));
    writeLabelMetric("Recall", meter.recall(labelId));
    std::cout << " " << dict->getLabel(labelId) << std::endl;
  }
}

int32_t neighborCount(
    const std::vector<std::string>& args,
    void (*printHelp)()) {
  if (args.size() == 3) {
    return 10;
  }
  if (args.size() == 4) {
    return std::stoi(args[3]);
  }
  printHelp();
  exit(EXIT_FAILURE);
}

} // namespace

void printTestUsage() {
  printEvaluationUsage("test", true);
}

void printPredictUsage() {
  printEvaluationUsage("predict[-prob]", true);
}

void printTestLabelUsage() {
  printEvaluationUsage("test-label", false);
}

void printPrintWordVectorsUsage() {
  std::cerr << "usage: fasttext print-word-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr << "usage: fasttext print-sentence-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr << "usage: fasttext print-ngrams <model> <word>\n\n"
            << "  <model>      model filename\n"
            << "  <word>       word to print\n"
            << std::endl;
}

void quantize(const std::vector<std::string>& args) {
  Args a = Args();
  if (args.size() < 3) {
    printQuantizeUsage();
    a.printHelp();
    exit(EXIT_FAILURE);
  }
  a.parseArgs(args);
  FastText fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a.output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel(a.output + ".ftz");
  exit(0);
}

void printNNUsage() {
  std::cout << "usage: fasttext nn <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printAnalogiesUsage() {
  std::cout << "usage: fasttext analogies <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printDumpUsage() {
  std::cout << "usage: fasttext dump <model> <option>\n\n"
            << "  <model>      model filename\n"
            << "  <option>     option from args,dict,input,output" << std::endl;
}

void test(const std::vector<std::string>& args) {
  bool perLabel = args[1] == "test-label";

  if (args.size() < 4 || args.size() > 6) {
    perLabel ? printTestLabelUsage() : printTestUsage();
    exit(EXIT_FAILURE);
  }

  const auto& model = args[2];
  const auto& input = args[3];
  const auto options = predictionOptions(args);
  const int32_t k = options.first;
  const real threshold = options.second;

  FastText fasttext;
  fasttext.loadModel(model);
  Meter meter(false);
  std::ifstream file;
  auto& in = openInput(input, file, "Test file cannot be opened!");
  fasttext.test(in, k, threshold, meter);
  if (file.is_open()) {
    file.close();
  }

  if (perLabel) {
    writeLabelMetrics(fasttext, meter);
  }
  meter.writeGeneralMetrics(std::cout, k);

  exit(0);
}

void printPredictions(
    const std::vector<std::pair<real, std::string>>& predictions,
    bool printProb,
    bool multiline) {
  bool first = true;
  for (const auto& prediction : predictions) {
    if (!first && !multiline) {
      std::cout << " ";
    }
    first = false;
    std::cout << prediction.second;
    if (printProb) {
      std::cout << " " << prediction.first;
    }
    if (multiline) {
      std::cout << std::endl;
    }
  }
  if (!multiline) {
    std::cout << std::endl;
  }
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 6) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  const auto options = predictionOptions(args);
  const int32_t k = options.first;
  const real threshold = options.second;

  bool printProb = args[1] == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::ifstream ifs;
  auto& in = openInput(args[3], ifs, "Input file cannot be opened!");
  std::vector<std::pair<real, std::string>> predictions;
  while (fasttext.predictLine(in, predictions, k, threshold)) {
    printPredictions(predictions, printProb, false);
  }
  if (ifs.is_open()) {
    ifs.close();
  }

  exit(0);
}

void printWordVectors(const std::vector<std::string>& args) {
  if (args.size() != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string word;
  Vector vec(fasttext.getDimension());
  while (std::cin >> word) {
    fasttext.getWordVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
  exit(0);
}

void printSentenceVectors(const std::vector<std::string>& args) {
  if (args.size() != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  Vector svec(fasttext.getDimension());
  while (std::cin.peek() != EOF) {
    fasttext.getSentenceVector(std::cin, svec);
    // Don't print sentence
    std::cout << svec << std::endl;
  }
  exit(0);
}

void printNgrams(const std::vector<std::string>& args) {
  if (args.size() != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string word(args[3]);
  std::vector<std::pair<std::string, Vector>> ngramVectors =
      fasttext.getNgramVectors(word);

  for (const auto& ngramVector : ngramVectors) {
    std::cout << ngramVector.first << " " << ngramVector.second << std::endl;
  }

  exit(0);
}

void nn(const std::vector<std::string>& args) {
  const int32_t k = neighborCount(args, printNNUsage);
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string prompt("Query word? ");
  std::cout << prompt;

  std::string queryWord;
  while (std::cin >> queryWord) {
    printPredictions(fasttext.getNN(queryWord, k), true, true);
    std::cout << prompt;
  }
  exit(0);
}

void analogies(const std::vector<std::string>& args) {
  const int32_t k = neighborCount(args, printAnalogiesUsage);
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  FastText fasttext;
  std::string model(args[2]);
  std::cout << "Loading model " << model << std::endl;
  fasttext.loadModel(model);

  std::string prompt("Query triplet (A - B + C)? ");
  std::string wordA, wordB, wordC;
  std::cout << prompt;
  while (true) {
    std::cin >> wordA;
    std::cin >> wordB;
    std::cin >> wordC;
    printPredictions(fasttext.getAnalogies(k, wordA, wordB, wordC), true, true);

    std::cout << prompt;
  }
  exit(0);
}

void train(const std::vector<std::string>& args) {
  Args a = Args();
  a.parseArgs(args);
  std::shared_ptr<FastText> fasttext = std::make_shared<FastText>();
  std::string outputFileName;

  if (a.hasAutotune() &&
      a.getAutotuneModelSize() != Args::kUnlimitedModelSize) {
    outputFileName = a.output + ".ftz";
  } else {
    outputFileName = a.output + ".bin";
  }
  std::ofstream ofs(outputFileName);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        outputFileName + " cannot be opened for saving.");
  }
  ofs.close();
  if (a.hasAutotune()) {
    Autotune autotune(fasttext);
    autotune.train(a);
  } else {
    fasttext->train(a);
  }
  fasttext->saveModel(outputFileName);
  fasttext->saveVectors(a.output + ".vec");
  if (a.saveOutput) {
    fasttext->saveOutput(a.output + ".output");
  }
}

void dump(const std::vector<std::string>& args) {
  if (args.size() < 4) {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }

  std::string modelPath = args[2];
  std::string option = args[3];

  FastText fasttext;
  fasttext.loadModel(modelPath);
  if (option == "args") {
    fasttext.getArgs().dump(std::cout);
  } else if (option == "dict") {
    fasttext.getDictionary()->dump(std::cout);
  } else if (option == "input") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getInputMatrix()->dump(std::cout);
    }
  } else if (option == "output") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getOutputMatrix()->dump(std::cout);
    }
  } else {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }
}

int runCommand(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  using CommandHandler = void (*)(const std::vector<std::string>&);
  static const std::pair<const char*, CommandHandler> commands[] = {
      {"skipgram", train},
      {"cbow", train},
      {"supervised", train},
      {"test", test},
      {"test-label", test},
      {"quantize", quantize},
      {"print-word-vectors", printWordVectors},
      {"print-sentence-vectors", printSentenceVectors},
      {"print-ngrams", printNgrams},
      {"nn", nn},
      {"analogies", analogies},
      {"predict", predict},
      {"predict-prob", predict},
      {"dump", dump},
  };
  for (const auto& command : commands) {
    if (args[1] == command.first) {
      command.second(args);
      return 0;
    }
  }
  printUsage();
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  try {
    return runCommand(argc, argv);
  } catch (const Dictionary::TokenTooLongError& e) {
    std::cerr << "[INPUT REJECTED] " << e.what() << '\n';
    return 2;
  } catch (const DenseMatrix::EmbeddingBlockError& e) {
    std::cerr << "[INPUT REJECTED] " << e.what() << '\n';
    return 2;
  }
}
