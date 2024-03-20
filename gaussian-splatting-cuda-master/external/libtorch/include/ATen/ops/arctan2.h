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



#include <ATen/ops/arctan2_ops.h>

namespace at {


// aten::arctan2(Tensor self, Tensor other) -> Tensor
inline at::Tensor arctan2(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::arctan2::call(self, other);
}

// aten::arctan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & arctan2_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::arctan2_out::call(self, other, out);
}
// aten::arctan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & arctan2_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::arctan2_out::call(self, other, out);
}

}
