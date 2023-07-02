#pragma once

#include <chrono>
#include <string>

#include <logger/LogType.hpp>

struct alignas(64) LogLine {
  std::chrono::time_point<std::chrono::high_resolution_clock> time;
  LogType type;
  std::string message;
};
