#include <logger/LogFileWriter.hpp>
#include <iostream>

LogFileWriter::LogFileWriter(const std::string &path, LogContainer &container) : container(container) {
  file.open(path);
  thread = std::jthread([this](std::stop_token stoken) {
    std::cout<< "Log file writer started" << std::endl;
    while (!stoken.stop_requested()) {
      if (this->container.needToLog()) {
      auto line = this->container.getLine();
      file << line.time.time_since_epoch().count() << " " << line.type << " "
           << line.message;
    }
    }
  });
}

LogFileWriter::~LogFileWriter() {
  file.close();
}