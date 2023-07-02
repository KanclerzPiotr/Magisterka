#include <logger/LogContainer.hpp>

LogContainer::LogContainer() : lines(128) {}

LogContainer::LogContainer(size_t size) : lines(size) {}

bool LogContainer::putLine(LogLine line) {
  lines[started++ % lines.size()] = line;
  finished++;
  return true;
}

bool LogContainer::needToLog() { return written < finished; }

LogLine LogContainer::getLine() {
  return lines[written++ % lines.size()];
}