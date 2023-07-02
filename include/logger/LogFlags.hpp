#pragma once
#include <ostream>

struct logEndl {
    char newLine = '\n';
};

inline std::ostream &operator<<(std::ostream &os, const logEndl &endl) {
  os << endl.newLine;
  return os;
}