#pragma once

#include <fstream>
#include <string>
#include <thread>

#include <logger/LogWritter.hpp>
#include <logger/LogContainer.hpp>

class LogFileWriter : public LogWritter {
private:
  LogContainer &container;
  std::ofstream file;
  std::jthread thread;

public:
  LogFileWriter(const std::string &path, LogContainer &container);
  virtual ~LogFileWriter();
};
