#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "admission/admission.h"
#include "core/config.h"
#include "core/storage_key.h"
#include "engine.h"

namespace py = pybind11;

using namespace attentiondb;

static void check_status(Status s, const char* op) {
    if (s == Status::kOk) return;
    throw std::runtime_error(std::string(op) + " failed: " +
                             std::string(StatusToString(s)));
}

class PyEngine {
public:
    explicit PyEngine(const std::string& config_path) {
        Config cfg;
        if (!config_path.empty()) {
            cfg = Config::LoadFromFile(config_path);
        }
        engine_ = std::make_unique<AttentionDBEngine>();
        check_status(engine_->open(cfg), "open");
    }

    explicit PyEngine(const Config& cfg) {
        engine_ = std::make_unique<AttentionDBEngine>();
        check_status(engine_->open(cfg), "open");
    }

    void put(const StorageKey& key, py::bytes data, const PutOpts& opts) {
        std::string buf = data;
        check_status(engine_->put(key, buf.data(), buf.size(), opts), "put");
    }

    std::optional<py::bytes> get(const StorageKey& key) {
        // Allocate a large buffer for the result
        std::vector<uint8_t> buf(4 * 1024 * 1024);  // 4MB max
        size_t out_len = 0;
        Status s = engine_->get(key, buf.data(), buf.size(), &out_len);

        if (s == Status::kNotFound) return std::nullopt;
        if (s == Status::kBufferTooSmall) {
            buf.resize(16 * 1024 * 1024);  // Retry with 16MB
            s = engine_->get(key, buf.data(), buf.size(), &out_len);
        }
        check_status(s, "get");
        return py::bytes(reinterpret_cast<const char*>(buf.data()), out_len);
    }

    bool contains(const StorageKey& key) {
        return engine_->contains(key);
    }

    std::vector<bool> batched_contains(const std::vector<StorageKey>& keys) {
        std::vector<bool> results(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = engine_->contains(keys[i]);
        }
        return results;
    }

    void del(const StorageKey& key) {
        engine_->del(key);
    }

    EngineStats stats() {
        return engine_->stats();
    }

    void close() {
        if (engine_) engine_->close();
    }

private:
    std::unique_ptr<AttentionDBEngine> engine_;
};

PYBIND11_MODULE(_attentiondb, m) {
    m.doc() = "AttentionDB storage engine Python bindings";

    py::class_<StorageKey>(m, "StorageKey")
        .def(py::init<>())
        .def(py::init([](uint64_t model_id, uint64_t tenant_id,
                         uint64_t chunk_hash, uint16_t layer_group_id,
                         uint32_t chunk_index) {
            StorageKey k;
            k.model_id = model_id;
            k.tenant_id = tenant_id;
            k.chunk_hash = chunk_hash;
            k.layer_group_id = layer_group_id;
            k.chunk_index = chunk_index;
            return k;
        }), py::arg("model_id"), py::arg("tenant_id"),
            py::arg("chunk_hash"), py::arg("layer_group_id"),
            py::arg("chunk_index"))
        .def_readwrite("model_id", &StorageKey::model_id)
        .def_readwrite("tenant_id", &StorageKey::tenant_id)
        .def_readwrite("chunk_hash", &StorageKey::chunk_hash)
        .def_readwrite("layer_group_id", &StorageKey::layer_group_id)
        .def_readwrite("chunk_index", &StorageKey::chunk_index)
        .def("__eq__", &StorageKey::operator==)
        .def("__hash__", [](const StorageKey& k) {
            return StorageKeyHash{}(k);
        });

    py::class_<PutOpts>(m, "PutOpts")
        .def(py::init<>())
        .def(py::init([](uint32_t num_tokens, uint32_t recompute_cost,
                         uint8_t entry_type, uint32_t ttl_seconds) {
            PutOpts o;
            o.num_tokens = num_tokens;
            o.recompute_cost = recompute_cost;
            o.entry_type = static_cast<EntryType>(entry_type);
            o.ttl_seconds = ttl_seconds;
            return o;
        }), py::arg("num_tokens") = 0, py::arg("recompute_cost") = 0,
            py::arg("entry_type") = 0, py::arg("ttl_seconds") = 0)
        .def_readwrite("num_tokens", &PutOpts::num_tokens)
        .def_readwrite("recompute_cost", &PutOpts::recompute_cost)
        .def_readwrite("ttl_seconds", &PutOpts::ttl_seconds);

    py::class_<EngineStats>(m, "Stats")
        .def_readonly("t1_total_bytes", &EngineStats::t1_total_bytes)
        .def_readonly("t1_used_bytes", &EngineStats::t1_used_bytes)
        .def_readonly("t2_total_bytes_on_disk", &EngineStats::t2_total_bytes_on_disk)
        .def_readonly("t2_num_segments", &EngineStats::t2_num_segments)
        .def_readonly("index_entries", &EngineStats::index_entries)
        .def_readonly("eviction_protected", &EngineStats::eviction_protected)
        .def_readonly("eviction_probationary", &EngineStats::eviction_probationary)
        .def_readonly("admission_evaluated", &EngineStats::admission_evaluated)
        .def_readonly("admission_rejected", &EngineStats::admission_rejected)
        .def_readonly("wb_submitted", &EngineStats::wb_submitted)
        .def_readonly("wb_rejected", &EngineStats::wb_rejected)
        .def_readonly("wb_flushed", &EngineStats::wb_flushed)
        .def_readonly("wb_utilization", &EngineStats::wb_utilization);

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<const std::string&>(), py::arg("config_path") = "")
        .def("put", &PyEngine::put, py::arg("key"), py::arg("data"), py::arg("opts"))
        .def("get", &PyEngine::get, py::arg("key"))
        .def("contains", &PyEngine::contains, py::arg("key"))
        .def("batched_contains", &PyEngine::batched_contains, py::arg("keys"))
        .def("delete", &PyEngine::del, py::arg("key"))
        .def("stats", &PyEngine::stats)
        .def("close", &PyEngine::close)
        .def("__enter__", [](PyEngine& self) -> PyEngine& { return self; })
        .def("__exit__", [](PyEngine& self, py::object, py::object, py::object) {
            self.close();
        });

    // Convenience function
    m.def("open", [](const std::string& config_path) {
        return std::make_unique<PyEngine>(config_path);
    }, py::arg("config_path") = "");
}
