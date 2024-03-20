#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <limits>
#include <memory>

namespace c10 {

// NB: this is actually double precision; we're using the Python naming here
class C10_API SymFloat {
 public:
  /*implicit*/ SymFloat(double d) : data_(d){};
  SymFloat(SymNode ptr)
      : data_(std::numeric_limits<double>::quiet_NaN()), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_float());
  };
  SymFloat() : data_(0.0) {}

  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  SymNode toSymNodeImpl() const;

  double expect_float() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  SymFloat operator+(const SymFloat&) const;
  SymFloat operator-(const SymFloat&) const;
  SymFloat operator*(const SymFloat&) const;
  SymFloat operator/(const SymFloat&) const;

  // Need guidance on where to put this code
  SymFloat sqrt() const;

  // Insert a guard for the float to be its concrete value, and then return
  // that value.  This operation always works, even if the float is symbolic,
  // so long as we know what the underlying value is. Don't blindly put this
  // everywhere; you can cause overspecialization of PyTorch programs with
  // this method.
  //
  // It should be called as guard_float(__FILE__, __LINE__).  The file and line
  // number can be used to diagnose overspecialization.
  double guard_float(const char* file, int64_t line) const;

  // N.B. It's important to keep this definition in the header
  // as we expect if checks to be folded for mobile builds
  // where `is_symbolic` is always false
  C10_ALWAYS_INLINE bool is_symbolic() const {
    return ptr_;
  }

  double as_float_unchecked() const {
    return data_;
  }

 private:
  // TODO: optimize to union
  double data_;
  SymNode ptr_;
};

C10_API std::ostream& operator<<(std::ostream& os, const SymFloat& s);
} // namespace c10
