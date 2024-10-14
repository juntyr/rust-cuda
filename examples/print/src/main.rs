#![deny(clippy::pedantic)]

use print::{kernel, link, Action};

fn main() -> rust_cuda::deps::cust::error::CudaResult<()> {
    // Link the non-generic CUDA kernel
    struct KernelPtx;
    link! { impl kernel for KernelPtx }

    // Initialize the CUDA API
    rust_cuda::deps::cust::init(rust_cuda::deps::cust::CudaFlags::empty())?;

    // Get the first CUDA GPU device
    let device = rust_cuda::deps::cust::device::Device::get_device(0)?;

    // Create a CUDA context associated to this device
    let _context = rust_cuda::host::CudaDropWrapper::from(
        rust_cuda::deps::cust::context::Context::create_and_push(
            rust_cuda::deps::cust::context::ContextFlags::MAP_HOST
                | rust_cuda::deps::cust::context::ContextFlags::SCHED_AUTO,
            device,
        )?,
    );

    // Create a new CUDA stream to submit kernels to
    let mut stream =
        rust_cuda::host::CudaDropWrapper::from(rust_cuda::deps::cust::stream::Stream::new(
            rust_cuda::deps::cust::stream::StreamFlags::NON_BLOCKING,
            None,
        )?);

    // Create a new instance of the CUDA kernel and prepare the launch config
    let mut kernel = rust_cuda::kernel::TypedPtxKernel::<kernel>::new::<KernelPtx>(None);
    let config = rust_cuda::kernel::LaunchConfig {
        grid: rust_cuda::deps::cust::function::GridSize::x(1),
        block: rust_cuda::deps::cust::function::BlockSize::x(4),
        ptx_jit: false,
    };

    // Launch the CUDA kernel on the stream and synchronise to its completion
    rust_cuda::host::Stream::with(&mut stream, |stream| {
        println!("Launching print kernel ...");
        kernel.launch1(stream, &config, Action::Print)?;
        println!("Launching panic kernel ...");
        kernel.launch1(stream, &config, Action::Panic)?;
        println!("Launching alloc error kernel ...");
        kernel.launch1(stream, &config, Action::AllocError)
    })?;

    Ok(())
}
