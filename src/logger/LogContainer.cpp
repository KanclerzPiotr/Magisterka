#include <logger/LogContainer.hpp>

LogContainer::LogContainer() : lines(128) {}

LogContainer::LogContainer(size_t size) : lines(size) {}

void LogContainer::putLine(LogLine line) {
  lines[started++ % lines.size()] = line;
  finished++;
}

bool LogContainer::needToLog() { return written < finished; }

LogLine LogContainer::getLine() { return lines[written++ % lines.size()]; }