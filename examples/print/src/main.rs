#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]

extern crate alloc;

#[derive(rust_cuda::const_type_layout::TypeLayout)]
#[layout(crate = "rust_cuda::const_type_layout")]
#[repr(C)]
pub enum Action {
    Print,
    Panic,
    AllocError,
}

#[rust_cuda::common::kernel(pub use link_kernel! as impl Kernel<KernelArgs, KernelPtx> for Launcher)]
#[kernel(allow(ptx::local_memory_usage))]
pub fn kernel(#[kernel(pass = SafeDeviceCopy)] action: Action) {
    match action {
        Action::Print => rust_cuda::device::utils::println!("println! from CUDA kernel"),
        Action::Panic => panic!("panic! from CUDA kernel"),
        Action::AllocError => {
            ::alloc::alloc::handle_alloc_error(::core::alloc::Layout::new::<i8>())
        },
    }
}

#[cfg(not(target_os = "cuda"))]
fn main() -> rust_cuda::rustacuda::error::CudaResult<()> {
    // Link the non-generic CUDA kernel
    type Launcher = rust_cuda::host::SimpleKernelLauncher<dyn Kernel>;
    link_kernel!();

    // Initialize the CUDA API
    rust_cuda::rustacuda::init(rust_cuda::rustacuda::CudaFlags::empty())?;

    // Get the first CUDA GPU device
    let device = rust_cuda::rustacuda::device::Device::get_device(0)?;

    // Create a CUDA context associated to this device
    let _context = rust_cuda::host::CudaDropWrapper::from(
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

    // Create a new CUDA stream to submit kernels to
    let stream = rust_cuda::host::CudaDropWrapper::from(rust_cuda::rustacuda::stream::Stream::new(
        rust_cuda::rustacuda::stream::StreamFlags::NON_BLOCKING,
        None,
    )?);

    // Create a new launcher for the CUDA kernel
    let mut launcher = Launcher {
        kernel: <Launcher as KernelPtx>::new_kernel()?,
        config: rust_cuda::host::LaunchConfig {
            grid: rust_cuda::rustacuda::function::GridSize::x(1),
            block: rust_cuda::rustacuda::function::BlockSize::x(4),
            shared_memory_size: 0,
            ptx_jit: false,
        },
    };

    // Launch the CUDA kernel on the stream and synchronise to its completion
    println!("Launching print kernel ...");
    launcher.kernel(&stream, Action::Print).unwrap();
    println!("Launching panic kernel ...");
    launcher.kernel(&stream, Action::Panic).unwrap_err();
    println!("Launching alloc error kernel ...");
    launcher.kernel(&stream, Action::AllocError).unwrap_err();

    Ok(())
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(info: &::core::panic::PanicInfo) -> ! {
        // pretty format and print the panic message
        // but don't allow dynamic formatting or panic payload downcasting
        rust_cuda::device::utils::pretty_panic_handler(info, false, false)
    }

    #[alloc_error_handler]
    #[track_caller]
    fn alloc_error_handler(layout: ::core::alloc::Layout) -> ! {
        rust_cuda::device::utils::pretty_alloc_error_handler(layout)
    }
}
