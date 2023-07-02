#ifndef __LOGTYPE_H__
#define __LOGTYPE_H__

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
#endif // __LOGTYPE_H__