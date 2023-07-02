#pragma once

#include <memory>
#include <sstream>

#include <logger/LogContainer.hpp>
#include <logger/LogWritter.hpp>
#include <logger/LogFlags.hpp>

class Logger {
private:
  inline static LogContainer container;
  inline static std::unique_ptr<LogWritter> writter;
  LogType type;
  std::stringstream ss;

public:
  Logger();
  ~Logger() = default;
  Logger &operator<<(const LogType &type);
  Logger &operator<<(const logEndl &endl);
  template <typename T> Logger &operator<<(const T &message);
  

  static LogContainer &getContainer();
  static void setWriter(std::unique_ptr<LogWritter> writer);
};

template <typename T> Logger &Logger::operator<<(const T &message) {
  ss << message;
  return *this;
}
