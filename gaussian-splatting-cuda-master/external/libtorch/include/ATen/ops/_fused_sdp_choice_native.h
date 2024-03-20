#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API int64_t _fused_sdp_choice_cpp(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask={}, double dropout_p=0.0, bool is_causal=false);
TORCH_API int64_t _fused_sdp_choice_cuda(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask={}, double dropout_p=0.0, bool is_causal=false);
TORCH_API int64_t _fused_sdp_choice_meta(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask={}, double dropout_p=0.0, bool is_causal=false);
} // namespace native
} // namespace at
