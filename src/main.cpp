#include <iostream>

#include <logger/LogFileWriter.hpp>
#include <logger/Logger.hpp>

int main() {

    std::unique_ptr<LogWritter> logFileWriter = std::make_unique<LogFileWriter>("logs.txt", Logger::getContainer());
    Logger::setWriter(std::move(logFileWriter));


    std::vector<std::thread> threads(50);

    for(int i =0; i < 50; i ++) {
        threads[i] = std::thread([i](){
          Logger log;
            for(int j = 0; j < 1000; j ++) {
                log << LogType::INFO << "Hello from thread " << i << " " << j << logEndl();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    for(int i =0; i < 50; i ++) {
        threads[i].join();
    }

    return 0;

}