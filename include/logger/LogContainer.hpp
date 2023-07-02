#pragma once

#include <atomic>
#include <vector>

#include <logger/LogLine.hpp>

class LogContainer {
private:
  std::vector<LogLine> lines;
  std::atomic<size_t> started{0};
  std::atomic<size_t> finished{0};
  std::atomic<size_t> written{0};

public:
  LogContainer();
  LogContainer(size_t size);
  ~LogContainer() = default;

  bool needToLog();
  bool putLine(LogLine line);
  LogLine getLine();
};
