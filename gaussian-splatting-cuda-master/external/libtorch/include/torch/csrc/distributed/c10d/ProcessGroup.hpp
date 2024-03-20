#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Work.hpp>
// *************************************************************************
// PROCESS GROUP collective communication API IS BEING CHANGED BETWEEN
// versions 1.7 and 1.8.
// PLEASE DO NOT ADD ANY DEPENDENCIES.
// SEE RFC: https://github.com/pytorch/pytorch/issues/39662
// *************************************************************************

constexpr auto kProcessGroupDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

// ProcessGroup is a base class that captures collective and point to
// point communication in a fixed set of processes.
//
// The functions specified in the class below describe the API alone;
// implementations are provided in subclasses.
//
// Every function that performs I/O is executed asynchronously by a
// thread pool owned by the ProcessGroup (by default). They return an
// object that can be used to wait for completion or error.
//
// The ProcessGroup can instantiate subgroups with fewer or an equal
// number of members. Implementations must take care that multiple
// process groups can be used in parallel and synchronize accordingly.
//
// The ProcessGroup assumes a fixed set of processes. If the set
// changes, existing instances must be destructed and instantiation
// and initialization must start from scratch. For members of the
// process group to find each other (referred to as rendezvous from
// hereon)
//
class TORCH_API ProcessGroup : public torch::CustomClassHolder {
 public:
  // ProcessGroup Options is a base struct that defines the basic options
  // when constructing a ProcessGroup. Each ProcessGroup subclass should
  // extend this struct and define its options if it wants to provide more
  // config options (beyond basic ones defined here) to end user.
  struct TORCH_API Options : torch::CustomClassHolder {
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout)
        : timeout(timeout), backend(std::move(backend)) {}
    ~Options() override = default;

    std::chrono::milliseconds timeout;

    // backend name
    const std::string backend;
  };

  enum BackendType {
    UNDEFINED = 0,
    GLOO = 1,
    NCCL = 2,
    UCC = 3,
    MPI = 4,
    CUSTOM = 5,
  };

  // Not used, set for backwards compatibility and only used for TypeDef in
  // Ops.cpp
  explicit ProcessGroup(int rank, int size);

  explicit ProcessGroup(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options);
  ~ProcessGroup() override;

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  virtual const std::string getBackendName() const {
    return options_->backend;
  };

  BackendType getBackendType() const {
    return backendType_;
  };

  virtual void startCoalescing(c10::DeviceType deviceType) {
    // only nccl has implemented startCoalescing so only execute for nccl
    // backends
    if (getBackendType() == BackendType::NCCL) {
      getBackend(deviceType)->startCoalescing();
    }
  }

  virtual void endCoalescing(
      c10::DeviceType deviceType,
      std::vector<c10::intrusive_ptr<Work>>& reqs) {
    // only nccl has implemented startCoalescing so only execute for nccl
    // backends
    if (getBackendType() == BackendType::NCCL) {
      getBackend(deviceType)->endCoalescing(reqs);
    }
  }

  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::broadcast_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    at::TensorList,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    int64_t,
                    int64_t,
                    int64_t)>();
    // It's awakward to unbox the opts here and box them again in the custom C++
    // op. But it's also complicated to make opts as a CustomClassHolder. Leave
    // it as it is now.
    return std::get<1>(op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.rootRank,
        opts.rootTensor,
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::allreduce_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    at::TensorList,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    const c10::intrusive_ptr<::c10d::ReduceOp>&,
                    int64_t)>();

    return std::get<1>(op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::allreduce_coalesced_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const c10::intrusive_ptr<::c10d::ReduceOp>&,
                             int64_t)>();

    return op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),

        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::reduce_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const c10::intrusive_ptr<::c10d::ReduceOp>&,
                             int64_t,
                             int64_t,
                             int64_t)>();
    return op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),

        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        opts.rootRank,
        opts.rootTensor,
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::allgather_", "")
                         .typed<std::tuple<
                             std::vector<std::vector<at::Tensor>>,
                             c10::intrusive_ptr<Work>>(
                             const std::vector<std::vector<at::Tensor>>&,
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t)>();

    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.timeout.count()));
  }

  // Gathers a single tensor inputBuffer into a single buffer outputBuffer that
  // is interpreted as a contigious collection of size inputBuffer * WORLD_SIZE.
  // For implementers of ProcessGroup API and advanced users only.
  // Note: this function will be deprecated in near future.
  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::_allgather_base_", "")
            .typed<std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(
                at::Tensor&,
                at::Tensor&,
                const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();

    return std::get<1>(op.call(
        outputBuffer,
        inputBuffer,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this)));
  }

  // This function is deprecated and will be moved out of ProcessGroup to comms:
  // * do not add dependencies on this function,
  // * do not implement it in your ProcessGroup, implement _allgather_base
  //   instead.
  virtual c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::allgather_coalesced_", "")
            .typed<c10::intrusive_ptr<Work>(
                const std::vector<std::vector<at::Tensor>>&,
                const at::TensorList&,
                const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();

    return op.call(
        outputTensorLists,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this));
  }

  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::gather_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             const std::vector<std::vector<at::Tensor>>&,
                             const at::TensorList&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t,
                             int64_t)>();
    return op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.rootRank,
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::scatter_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    const at::TensorList&,
                    const std::vector<std::vector<at::Tensor>>&,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    int64_t,
                    int64_t)>();
    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.rootRank,
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::reduce_scatter_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    const at::TensorList&,
                    const std::vector<std::vector<at::Tensor>>&,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    const c10::intrusive_ptr<::c10d::ReduceOp>&,
                    int64_t)>();
    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::_reduce_scatter_base_", "")
                         .typed<std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(
                             at::Tensor&,
                             at::Tensor&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const c10::intrusive_ptr<::c10d::ReduceOp>&,
                             int64_t)>();
    return std::get<1>(op.call(
        outputBuffer,
        inputBuffer,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::alltoall_base_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::Tensor&,
                             at::Tensor&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             std::vector<int64_t>,
                             std::vector<int64_t>,
                             int64_t)>();
    return op.call(
        outputBuffer,
        inputBuffer,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        outputSplitSizes,
        inputSplitSizes,
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::alltoall_", "")
                         .typed<std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                             const at::TensorList&,
                             const at::TensorList&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t)>();
    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.timeout.count()));
  }

  virtual void monitoredBarrier(
      const BarrierOptions& opts,
      bool wait_all_ranks = false) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::monitored_barrier_", "")
                         .typed<void(
                             at::Tensor,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const std::vector<int64_t>&,
                             int64_t,
                             bool)>();
    // Default to using cpu implementation, monitored barrier is only for GLOO
    at::Tensor tensor = at::empty({0}, at::TensorOptions().device(at::kCPU));
    op.call(
        tensor,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.device_ids,
        opts.timeout.count(),
        wait_all_ranks);
  }

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store. Only implemented
  // for GLOO and NCCL backends currently.
  virtual void setSequenceNumberForGroup() {
    auto backendType = getBackendType();
    // TODO: HACK for backend name to get sequence number for that backend.
    if (backendType == ProcessGroup::BackendType::GLOO ||
        backendType == ProcessGroup::BackendType::NCCL ||
        backendType == ProcessGroup::BackendType::UCC) {
      getDefaultBackend()->setSequenceNumberForGroup();
    } else {
      TORCH_CHECK(
          false,
          c10::str(
              "ProcessGroup ",
              getBackendName(),
              " does not yet support sequence numbers."));
    }
  }

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  virtual uint64_t getSequenceNumberForGroup() {
    auto backendType = getBackendType();

    // TODO: HACK for backend name to get sequence number for that backend.
    if (backendType == ProcessGroup::BackendType::GLOO ||
        backendType == ProcessGroup::BackendType::NCCL ||
        backendType == ProcessGroup::BackendType::UCC) {
      return getDefaultBackend()->getSequenceNumberForGroup();
    } else {
      TORCH_CHECK(
          false,
          c10::str(
              "ProcessGroup ",
              getBackendName(),
              " does not yet support sequence numbers."));
    }
  }

  virtual c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::send", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t,
                             int64_t)>();
    return op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        dstRank,
        tag);
  }

  virtual c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::recv_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t,
                             int64_t)>();
    return op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        srcRank,
        tag);
  }

  virtual c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::recv_any_source_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t)>();
    return op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        tag);
  }

  virtual c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) {
    static at::Tensor tensor;
    // TODO: if nccl was specified then use it
    if (backendType_ == c10d::ProcessGroup::BackendType::NCCL) {
      // set cuda tensor
      tensor = at::empty(
          {1},
          at::TensorOptions().device(at::DeviceType::CUDA).dtype(at::kByte));
    } else {
      // Default to using cpu implementation
      tensor = at::empty(
          {1},
          at::TensorOptions().device(at::DeviceType::CPU).dtype(at::kByte));
    }

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::barrier", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::Tensor,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const std::vector<int64_t>&,
                             int64_t)>();

    return op.call(
        tensor,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.device_ids,
        opts.timeout.count());
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  bool hasBackends() {
    return !deviceTypeToBackendType_.empty();
  }

  void setBackend(
      c10::DeviceType deviceType,
      BackendType backendType,
      const c10::optional<c10::intrusive_ptr<Backend>>& backend) {
    deviceTypeToBackendType_[deviceType] = backendType;
    // if the backendType is already set then reuse it for this device
    if (backendTypeToBackend_.find(backendType) !=
        backendTypeToBackend_.end()) {
      auto existingBackend = backendTypeToBackend_.at(backendType);
      deviceTypeToBackend_[deviceType] = existingBackend;
    } else {
      // check if backend has value
      if (backend.has_value()) {
        deviceTypeToBackend_[deviceType] = backend.value();
        backendTypeToBackend_[backendType] = backend.value();
      }
    }
  }

  c10::intrusive_ptr<Backend> getDefaultBackend() const {
    TORCH_CHECK(
        backendTypeToBackend_.find(backendType_) != backendTypeToBackend_.end(),
        "Could not find the default backend type ",
        backendType_,
        " for Process Group with name ",
        getBackendName(),
        ".");
    return backendTypeToBackend_.at(backendType_);
  }

  c10::intrusive_ptr<Backend> getBackend(c10::DeviceType deviceType);

  c10::intrusive_ptr<Backend> getBackend(BackendType backendType) const {
    TORCH_CHECK(
        backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end(),
        "Could not find backend type ",
        backendType,
        ".");
    return backendTypeToBackend_.at(backendType);
  }

 protected:
  // Implementations of this interface need to call this to setup
  // appropriate logging etc.
  void init();

  const c10::intrusive_ptr<c10d::Store> store_;
  const int rank_;
  const int size_;
  const c10::intrusive_ptr<Options> options_;
  const BackendType backendType_;
  // Optional sequence number structure for matching collectives.
  c10::optional<c10d::SequenceNum> sequenceNum_ = c10::nullopt;

  // Debug level setting. It is parsed once when ProcessGroup is constructed and
  // remains the same across use of this process group.
  DebugLevel dist_debug_level_;

  // Backend classes for this ProcessGroup
  std::unordered_map<c10::DeviceType, BackendType> deviceTypeToBackendType_;
  std::unordered_map<c10::DeviceType, c10::intrusive_ptr<Backend>>
      deviceTypeToBackend_;
  std::unordered_map<BackendType, c10::intrusive_ptr<Backend>>
      backendTypeToBackend_;
};

} // namespace c10d
