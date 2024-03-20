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



#include <ATen/ops/_scaled_dot_product_efficient_attention_backward_ops.h>

namespace at {


// aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, bool is_causal=False, bool chunk_grad_outputs=False) -> (Tensor, Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, bool is_causal=false, bool chunk_grad_outputs=false) {
    return at::_ops::_scaled_dot_product_efficient_attention_backward::call(grad_out_, query, key, value, out, logsumexp, is_causal, chunk_grad_outputs);
}

}
