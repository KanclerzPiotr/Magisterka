#include <logger/Logger.hpp>
#include <iostream>

Logger::Logger() {}

Logger &Logger::operator<<(const LogType &t) {
  type = t;
  return *this;
}

Logger &Logger::operator<<(const logEndl &endl) {
  ss << endl;
  bool success = container.putLine(
      {std::chrono::high_resolution_clock::now(), type, ss.str()});
  if(!success) {
    std::cout<< "Log container is full" << std::endl;
  }
  ss.str("");
  return *this;
}

LogContainer &Logger::getContainer() { return container; }

void Logger::setWriter(std::unique_ptr<LogWritter> writer) {
  writter = std::move(writer);
}
