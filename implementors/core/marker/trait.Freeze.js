(function() {var implementors = {};
implementors["rust_cuda"] = [{"text":"impl&lt;T:&nbsp;?<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/common/struct.DeviceAccessible.html\" title=\"struct rust_cuda::common::DeviceAccessible\">DeviceAccessible</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::common::DeviceAccessible"]},{"text":"impl&lt;'r, T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/common/struct.DeviceConstRef.html\" title=\"struct rust_cuda::common::DeviceConstRef\">DeviceConstRef</a>&lt;'r, T&gt;","synthetic":true,"types":["rust_cuda::common::DeviceConstRef"]},{"text":"impl&lt;'r, T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/common/struct.DeviceMutRef.html\" title=\"struct rust_cuda::common::DeviceMutRef\">DeviceMutRef</a>&lt;'r, T&gt;","synthetic":true,"types":["rust_cuda::common::DeviceMutRef"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.LaunchConfig.html\" title=\"struct rust_cuda::host::LaunchConfig\">LaunchConfig</a>","synthetic":true,"types":["rust_cuda::host::LaunchConfig"]},{"text":"impl&lt;'l, L:&nbsp;?<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.LaunchPackage.html\" title=\"struct rust_cuda::host::LaunchPackage\">LaunchPackage</a>&lt;'l, L&gt;","synthetic":true,"types":["rust_cuda::host::LaunchPackage"]},{"text":"impl&lt;'k&gt; Freeze for <a class=\"enum\" href=\"rust_cuda/host/enum.KernelJITResult.html\" title=\"enum rust_cuda::host::KernelJITResult\">KernelJITResult</a>&lt;'k&gt;","synthetic":true,"types":["rust_cuda::host::KernelJITResult"]},{"text":"impl&lt;KernelTraitObject:&nbsp;?<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.TypedKernel.html\" title=\"struct rust_cuda::host::TypedKernel\">TypedKernel</a>&lt;KernelTraitObject&gt;","synthetic":true,"types":["rust_cuda::host::TypedKernel"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.NullCudaAlloc.html\" title=\"struct rust_cuda::host::NullCudaAlloc\">NullCudaAlloc</a>","synthetic":true,"types":["rust_cuda::host::NullCudaAlloc"]},{"text":"impl&lt;A, B&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.CombinedCudaAlloc.html\" title=\"struct rust_cuda::host::CombinedCudaAlloc\">CombinedCudaAlloc</a>&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Freeze,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::host::CombinedCudaAlloc"]},{"text":"impl&lt;C&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.CudaDropWrapper.html\" title=\"struct rust_cuda::host::CudaDropWrapper\">CudaDropWrapper</a>&lt;C&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;C: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::host::CudaDropWrapper"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.HostDeviceBox.html\" title=\"struct rust_cuda::host::HostDeviceBox\">HostDeviceBox</a>&lt;T&gt;","synthetic":true,"types":["rust_cuda::host::HostDeviceBox"]},{"text":"impl&lt;'a, T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.HostAndDeviceMutRef.html\" title=\"struct rust_cuda::host::HostAndDeviceMutRef\">HostAndDeviceMutRef</a>&lt;'a, T&gt;","synthetic":true,"types":["rust_cuda::host::HostAndDeviceMutRef"]},{"text":"impl&lt;'a, T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.HostAndDeviceConstRef.html\" title=\"struct rust_cuda::host::HostAndDeviceConstRef\">HostAndDeviceConstRef</a>&lt;'a, T&gt;","synthetic":true,"types":["rust_cuda::host::HostAndDeviceConstRef"]},{"text":"impl&lt;'a, T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/host/struct.HostAndDeviceOwned.html\" title=\"struct rust_cuda::host::HostAndDeviceOwned\">HostAndDeviceOwned</a>&lt;'a, T&gt;","synthetic":true,"types":["rust_cuda::host::HostAndDeviceOwned"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/device/utils/struct.PTXAllocator.html\" title=\"struct rust_cuda::device::utils::PTXAllocator\">PTXAllocator</a>","synthetic":true,"types":["rust_cuda::device::utils::PTXAllocator"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/device/utils/struct.Dim3.html\" title=\"struct rust_cuda::device::utils::Dim3\">Dim3</a>","synthetic":true,"types":["rust_cuda::device::utils::Dim3"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/device/utils/struct.Idx3.html\" title=\"struct rust_cuda::device::utils::Idx3\">Idx3</a>","synthetic":true,"types":["rust_cuda::device::utils::Idx3"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/device/struct.ShallowCopy.html\" title=\"struct rust_cuda::device::ShallowCopy\">ShallowCopy</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::device::ShallowCopy"]},{"text":"impl&lt;T, const STRIDE:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/aliasing/struct.SplitSliceOverCudaThreadsConstStride.html\" title=\"struct rust_cuda::utils::aliasing::SplitSliceOverCudaThreadsConstStride\">SplitSliceOverCudaThreadsConstStride</a>&lt;T, STRIDE&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::aliasing::const::SplitSliceOverCudaThreadsConstStride"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/aliasing/struct.SplitSliceOverCudaThreadsDynamicStride.html\" title=\"struct rust_cuda::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride\">SplitSliceOverCudaThreadsDynamicStride</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::aliasing::dynamic::SplitSliceOverCudaThreadsDynamicStride"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda/utils/alloc/struct.UnifiedAllocator.html\" title=\"struct rust_cuda::utils::alloc::UnifiedAllocator\">UnifiedAllocator</a>","synthetic":true,"types":["rust_cuda::utils::alloc::UnifiedAllocator"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/device_copy/struct.SafeDeviceCopyWrapper.html\" title=\"struct rust_cuda::utils::device_copy::SafeDeviceCopyWrapper\">SafeDeviceCopyWrapper</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::device_copy::SafeDeviceCopyWrapper"]},{"text":"impl&lt;T, const M2D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>, const M2H:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/exchange/buffer/struct.CudaExchangeBufferDevice.html\" title=\"struct rust_cuda::utils::exchange::buffer::CudaExchangeBufferDevice\">CudaExchangeBufferDevice</a>&lt;T, M2D, M2H&gt;","synthetic":true,"types":["rust_cuda::utils::exchange::buffer::device::CudaExchangeBufferDevice"]},{"text":"impl&lt;T, const M2D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>, const M2H:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>&gt; !Freeze for <a class=\"struct\" href=\"rust_cuda/utils/exchange/buffer/struct.CudaExchangeBuffer.html\" title=\"struct rust_cuda::utils::exchange::buffer::CudaExchangeBuffer\">CudaExchangeBufferHost</a>&lt;T, M2D, M2H&gt;","synthetic":true,"types":["rust_cuda::utils::exchange::buffer::host::CudaExchangeBufferHost"]},{"text":"impl&lt;T, const M2D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>, const M2H:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a>&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/exchange/buffer/struct.CudaExchangeItem.html\" title=\"struct rust_cuda::utils::exchange::buffer::CudaExchangeItem\">CudaExchangeItem</a>&lt;T, M2D, M2H&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::exchange::buffer::CudaExchangeItem"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/exchange/wrapper/struct.ExchangeWrapperOnHost.html\" title=\"struct rust_cuda::utils::exchange::wrapper::ExchangeWrapperOnHost\">ExchangeWrapperOnHost</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::exchange::wrapper::ExchangeWrapperOnHost"]},{"text":"impl&lt;T&gt; Freeze for <a class=\"struct\" href=\"rust_cuda/utils/exchange/wrapper/struct.ExchangeWrapperOnDevice.html\" title=\"struct rust_cuda::utils::exchange::wrapper::ExchangeWrapperOnDevice\">ExchangeWrapperOnDevice</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Freeze,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;T as <a class=\"trait\" href=\"rust_cuda/common/trait.RustToCuda.html\" title=\"trait rust_cuda::common::RustToCuda\">RustToCuda</a>&gt;::<a class=\"associatedtype\" href=\"rust_cuda/common/trait.RustToCuda.html#associatedtype.CudaAllocation\" title=\"type rust_cuda::common::RustToCuda::CudaAllocation\">CudaAllocation</a>: Freeze,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;T as <a class=\"trait\" href=\"rust_cuda/common/trait.RustToCuda.html\" title=\"trait rust_cuda::common::RustToCuda\">RustToCuda</a>&gt;::<a class=\"associatedtype\" href=\"rust_cuda/common/trait.RustToCuda.html#associatedtype.CudaRepresentation\" title=\"type rust_cuda::common::RustToCuda::CudaRepresentation\">CudaRepresentation</a>: Freeze,&nbsp;</span>","synthetic":true,"types":["rust_cuda::utils::exchange::wrapper::ExchangeWrapperOnDevice"]}];
implementors["rust_cuda_ptx_jit"] = [{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda_ptx_jit/struct.PtxJITCompiler.html\" title=\"struct rust_cuda_ptx_jit::PtxJITCompiler\">PtxJITCompiler</a>","synthetic":true,"types":["rust_cuda_ptx_jit::host::compiler::PtxJITCompiler"]},{"text":"impl&lt;'s&gt; Freeze for <a class=\"enum\" href=\"rust_cuda_ptx_jit/enum.PtxJITResult.html\" title=\"enum rust_cuda_ptx_jit::PtxJITResult\">PtxJITResult</a>&lt;'s&gt;","synthetic":true,"types":["rust_cuda_ptx_jit::host::compiler::PtxJITResult"]},{"text":"impl Freeze for <a class=\"struct\" href=\"rust_cuda_ptx_jit/struct.CudaKernel.html\" title=\"struct rust_cuda_ptx_jit::CudaKernel\">CudaKernel</a>","synthetic":true,"types":["rust_cuda_ptx_jit::host::kernel::CudaKernel"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()