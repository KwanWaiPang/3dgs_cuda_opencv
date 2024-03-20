#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace impl {
namespace dispatch {

void initDispatchBindings(PyObject* module);

void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    torch::jit::Stack* stack);

} // namespace dispatch
} // namespace impl
} // namespace torch
