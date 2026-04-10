// Shim: c_api symbols are compiled into attentiondb_core (static lib).
// This TU exists solely so CMake has a source file for the shared library target.
// All public C API symbols are re-exported via the attentiondb_core linkage.
