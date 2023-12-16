#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(ptr_from_ref)]
#![feature(stdsimd)]
#![feature(c_str_literals)]

extern crate alloc;

#[cfg(not(target_os = "cuda"))]
fn main() -> rust_cuda::rustacuda::error::CudaResult<()> {
    // Initialize the CUDA API
    rust_cuda::rustacuda::init(rust_cuda::rustacuda::CudaFlags::empty())?;

    // Get the first device
    let device = rust_cuda::rustacuda::device::Device::get_device(0)?;

    // Create a context associated to this device
    let context = rust_cuda::host::CudaDropWrapper::from(
        rust_cuda::rustacuda::context::Context::create_and_push(
            rust_cuda::rustacuda::context::ContextFlags::MAP_HOST
                | rust_cuda::rustacuda::context::ContextFlags::SCHED_AUTO,
            device,
        )?,
    );

    rust_cuda::rustacuda::context::CurrentContext::set_resource_limit(
        rust_cuda::rustacuda::context::ResourceLimit::StackSize,
        4096,
    )?;
    rust_cuda::rustacuda::context::CurrentContext::set_resource_limit(
        rust_cuda::rustacuda::context::ResourceLimit::PrintfFifoSize,
        4096,
    )?;

    let stream = rust_cuda::host::CudaDropWrapper::from(rust_cuda::rustacuda::stream::Stream::new(
        rust_cuda::rustacuda::stream::StreamFlags::NON_BLOCKING,
        None,
    )?);

    let mut kernel = host::Launcher::try_new(
        rust_cuda::rustacuda::function::GridSize::x(1),
        rust_cuda::rustacuda::function::BlockSize::x(4),
    )?;

    kernel.kernel(&stream)?;

    std::mem::drop(context);

    Ok(())
}

#[rust_cuda::common::kernel(pub use link_kernel! as impl Kernel<KernelArgs, KernelPtx> for Launcher)]
#[kernel(allow(ptx::local_memory_usage))]
pub fn kernel() {
    rust_cuda::device::utils::print(format_args!("print from CUDA kernel\n"));

    ::alloc::alloc::handle_alloc_error(::core::alloc::Layout::new::<i8>());
}

#[cfg(not(target_os = "cuda"))]
mod host {
    #[allow(unused_imports)]
    use super::KernelArgs;
    use super::{Kernel, KernelPtx};

    pub struct Launcher {
        kernel: rust_cuda::host::TypedKernel<dyn Kernel>,
        grid: rust_cuda::rustacuda::function::GridSize,
        block: rust_cuda::rustacuda::function::BlockSize,
        watcher: (),
    }

    impl Launcher {
        pub fn try_new(
            grid: rust_cuda::rustacuda::function::GridSize,
            block: rust_cuda::rustacuda::function::BlockSize,
        ) -> rust_cuda::rustacuda::error::CudaResult<Self> {
            let kernel = Self::new_kernel()?;

            Ok(Self {
                kernel,
                grid,
                block,
                watcher: (),
            })
        }
    }

    link_kernel!();

    impl rust_cuda::host::Launcher for Launcher {
        type CompilationWatcher = ();
        type KernelTraitObject = dyn Kernel;

        fn get_launch_package(&mut self) -> rust_cuda::host::LaunchPackage<Self> {
            rust_cuda::host::LaunchPackage {
                config: rust_cuda::host::LaunchConfig {
                    grid: self.grid.clone(),
                    block: self.block.clone(),
                    shared_memory_size: 0_u32,
                    ptx_jit: false,
                },

                kernel: &mut self.kernel,

                watcher: &mut self.watcher,
            }
        }
    }
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(info: &::core::panic::PanicInfo) -> ! {
        rust_cuda::device::utils::print(format_args!("{info}\n"));

        rust_cuda::device::utils::abort()
    }

    #[alloc_error_handler]
    #[track_caller]
    fn alloc_error_handler(layout: ::core::alloc::Layout) -> ! {
        #[repr(C)]
        struct FormatArgs {
            size: usize,
            align: usize,
            thread_idx_x: u32,
            thread_idx_y: u32,
            thread_idx_z: u32,
            file_len: u32,
            file_ptr: *const u8,
            line: u32,
            column: u32,
        }

        let thread_idx = rust_cuda::device::thread::Thread::this().idx();
        let location = ::core::panic::Location::caller();

        unsafe {
            ::core::arch::nvptx::vprintf(
                c"memory allocation of %llu bytes with alignment %llu failed on thread (x=%u, y=%u, z=%u) at %*s:%u:%u\n"
                    .as_ptr()
                    .cast(),
                #[allow(clippy::cast_possible_truncation)]
                ::core::ptr::from_ref(&FormatArgs {
                    size: layout.size(),
                    align: layout.align(),
                    thread_idx_x: thread_idx.x,
                    thread_idx_y: thread_idx.y,
                    thread_idx_z: thread_idx.z,
                    file_len: location.file().len() as u32,
                    file_ptr: location.file().as_ptr(),
                    line: location.line(),
                    column: location.column(),
                })
                .cast(),
            );
        }

        rust_cuda::device::utils::abort()
    }
}
