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



#include <ATen/ops/_fused_dropout_ops.h>

namespace at {


// aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor> _fused_dropout(const at::Tensor & self, double p, c10::optional<at::Generator> generator=c10::nullopt) {
    return at::_ops::_fused_dropout::call(self, p, generator);
}

// aten::_fused_dropout.out(Tensor self, float p, Generator? generator=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
inline ::std::tuple<at::Tensor &,at::Tensor &> _fused_dropout_out(at::Tensor & out0, at::Tensor & out1, const at::Tensor & self, double p, c10::optional<at::Generator> generator=c10::nullopt) {
    return at::_ops::_fused_dropout_out::call(self, p, generator, out0, out1);
}
// aten::_fused_dropout.out(Tensor self, float p, Generator? generator=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
inline ::std::tuple<at::Tensor &,at::Tensor &> _fused_dropout_outf(const at::Tensor & self, double p, c10::optional<at::Generator> generator, at::Tensor & out0, at::Tensor & out1) {
    return at::_ops::_fused_dropout_out::call(self, p, generator, out0, out1);
}

}
