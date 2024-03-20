#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/divide_ops.h>

namespace at {


// aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
inline at::Tensor divide(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::divide_Tensor::call(self, other);
}

// aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & divide_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::divide_out::call(self, other, out);
}
// aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & divide_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::divide_out::call(self, other, out);
}

// aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
inline at::Tensor divide(const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::divide_Scalar::call(self, other);
}

// aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
inline at::Tensor divide(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    return at::_ops::divide_Tensor_mode::call(self, other, rounding_mode);
}

// aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & divide_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    return at::_ops::divide_out_mode::call(self, other, rounding_mode, out);
}
// aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & divide_outf(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
    return at::_ops::divide_out_mode::call(self, other, rounding_mode, out);
}

// aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
inline at::Tensor divide(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    return at::_ops::divide_Scalar_mode::call(self, other, rounding_mode);
}

}
