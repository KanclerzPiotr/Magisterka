#include <logger/Logger.hpp>

Logger::Logger() {}

Logger &Logger::operator<<(const LogType &t) {
  type = t;
  return *this;
}

Logger &Logger::operator<<(const logEndl &endl) {
  ss << endl;
  container.putLine(
      {std::chrono::high_resolution_clock::now(), type, ss.str()});
  ss.str("");
  return *this;
}

LogContainer &Logger::getContainer() { return container; }

void Logger::setWriter(std::unique_ptr<LogWritter> writer) {
  writter = std::move(writer);
}
