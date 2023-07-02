#pragma once

#include <ostream>

enum class LogType { INFO, DEBUG, WARN, ERROR };

inline std::ostream &operator<<(std::ostream &os, const LogType &type) {
  switch (type) {
  case LogType::INFO:
    os << "INFO";
    break;
  case LogType::DEBUG:
    os << "DEBUG";
    break;
  case LogType::WARN:
    os << "WARN";
    break;
  case LogType::ERROR:
    os << "ERROR";
    break;
  }
  return os;
}