#pragma once

#include <cstdint>
#include <string_view>

namespace attentiondb {

enum class Status : uint8_t {
    kOk = 0,
    kNotFound = 1,
    kRejectedAdmission = 2,
    kRejectedBackpressure = 3,
    kFull = 4,
    kBufferTooSmall = 5,
    kCorruption = 6,
    kIOError = 7,
    kInvalidArgument = 8,
    kAlreadyExists = 9,
    kShutdown = 10,
};

constexpr std::string_view StatusToString(Status s) {
    switch (s) {
        case Status::kOk:                   return "OK";
        case Status::kNotFound:             return "NOT_FOUND";
        case Status::kRejectedAdmission:    return "REJECTED_ADMISSION";
        case Status::kRejectedBackpressure: return "REJECTED_BACKPRESSURE";
        case Status::kFull:                 return "FULL";
        case Status::kBufferTooSmall:       return "BUFFER_TOO_SMALL";
        case Status::kCorruption:           return "CORRUPTION";
        case Status::kIOError:              return "IO_ERROR";
        case Status::kInvalidArgument:      return "INVALID_ARGUMENT";
        case Status::kAlreadyExists:        return "ALREADY_EXISTS";
        case Status::kShutdown:             return "SHUTDOWN";
    }
    return "UNKNOWN";
}

}  // namespace attentiondb
