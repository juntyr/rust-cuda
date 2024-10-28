//! [![CI Status]][workflow] [![MSRV]][repo] [![Rust Doc]][docs] [![License
//! Status]][fossa] [![Code Coverage]][codecov] [![Gitpod
//! Ready-to-Code]][gitpod]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/rust-cuda/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/rust-cuda/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.81.0--nightly-orange
//! [repo]: https://github.com/juntyr/rust-cuda
//!
//! [Rust Doc]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/rust-cuda/rust_cuda_kernel/
//!
//! [License Status]: https://app.fossa.com/api/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda.svg?type=shield
//! [fossa]: https://app.fossa.com/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda?ref=badge_shield
//!
//! [Code Coverage]: https://img.shields.io/codecov/c/github/juntyr/rust-cuda?token=wfeAeybbbx
//! [codecov]: https://codecov.io/gh/juntyr/rust-cuda
//!
//! [Gitpod Ready-to-Code]: https://img.shields.io/badge/Gitpod-ready-blue?logo=gitpod
//! [gitpod]: https://gitpod.io/#https://github.com/juntyr/rust-cuda
//!
//! `rust-cuda-kernel` provides the [`#[kernel]`](macro@kernel) attribute
//! macro. When applied to a function, it compiles it as a CUDA kernel that
//! can be *safely* called from Rust code on the host.

#![deny(unsafe_code)]
#![feature(box_patterns)]
#![feature(proc_macro_tracked_env)]
#![feature(proc_macro_span)]
#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(proc_macro_def_site)]
#![feature(cfg_version)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error2;

use proc_macro::TokenStream;

mod kernel;

#[proc_macro_error]
#[proc_macro_attribute]
/// Provides the [`#[kernel]`](macro@kernel) attribute macro. When applied to a
/// function, it compiles it as a CUDA kernel that can be *safely* called from
/// Rust code on the host.
///
/// The annotated function must be public, not const, not async, not have an
/// explicit ABI, not be variadic, not have a receiver (e.g. `&self`), and
/// return the unit type `()`. At the moment, the kernel function must also
/// not use a where clause – use type generic bounds instead.
///
/// While the [`#[kernel]`](macro@kernel) attribute supports functions with any
/// number of arguments, [`rust_cuda::kernel::TypedPtxKernel`] only supports
/// launching kernels with up to 12 parameters at the moment.
///
/// The [`#[kernel]`](macro@kernel) attribute uses the following syntax:
///
/// ```rust,ignore
/// #[kernel(pub? use link! for impl)]
/// fn my_kernel(/* parameters */) {
///     /* kernel code */
/// }
/// ```
///
/// where `link` is the name of a macro that will be generated to manually link
/// specific monomorphised instantiations of the (optionally generic) kernel
/// function, and the optional `pub` controls whether this macro is public or
/// private.
///
/// Note that all kernel parameters must implement the sealed
/// [`rust_cuda::kernel::CudaKernelParameter`] trait.
///
/// To use a specific monomorphised instantiation of the kernel, the generated
/// `link!` macro must be invoked with the following syntax:
///
/// ```rust,ignore
/// struct KernelPtx;
/// link! { impl my_kernel for KernelPtx }
/// ```
/// for the non-generic kernel function `my_kernel` and a non-generic marker
/// type `KernelPtx`, which can be used as the generic `Kernel` type parameter
/// for [`rust_cuda::kernel::TypedPtxKernel`] to instantiate and launch the
/// kernel. Specifically, the [`rust_cuda::kernel::CompiledKernelPtx`] trait is
/// implemented for the `KernelPtx` type.
///
/// If the kernel function is generic, the following syntax is used instead:
/// ```rust,ignore
/// #[kernel(pub? use link! for impl)]
/// fn my_kernel<'a, A, B: Bounded, const N: usize>(/* parameters */) {
///     /* kernel code */
/// }
///
/// struct KernelPtx<'a, A, B: Bounded, const N: usize>(/* ... */);
/// link! { impl my_kernel<'a, u32, MyStruct, 42> for KernelPtx }
/// link! { impl my_kernel<'a, bool, MyOtherStruct, 24> for KernelPtx }
/// ```
///
/// If the kernel generic space is closed, the `link!` macro can be made
/// private and all instantiations must be requested in the same crate that
/// defines the kernel function. If downstream code should be allowed to use
/// and compile new specific monomorphised instantiations of the kernel, the
/// `link!` macro should be publicly exported. Then, downstream code can define
/// its own `MyKernelPtx` marker types for which the kernel is linked and which
/// can be passed to [`rust_cuda::kernel::CompiledKernelPtx`]-generic code in
/// the kernel-defining crate to construct the requested
/// [`rust_cuda::kernel::TypedPtxKernel`].
///
/// Inside the scope of the [`#[kernel]`](macro@kernel) attribute, a helper
/// `#[kernel(...)]` attribute can be applied to the kernel function:
///
/// - `#[kernel(crate = "<crate-path>")]` changes the path to the [`rust-cuda`]
///   crate that the kernel compilation uses, which by default is `rust_cuda`.
/// - `#[kernel(allow/warn/deny/forbid(<lint>))]` checks the specified
///   CUDA-specific lint for each kernel compilation, using default Rust
///   semantics for allowing, warning on, denying, or forbidding a lint. The
///   following lints are supported:
///   - `ptx::double_precision_use`: check for any uses of [`f64`] operations
///     inside the compiled PTX binary, as they are often significantly less
///     performant on NVIDIA GPUs than [`f32`] operations. By default,
///     `#[kernel(warn(ptx::double_precision_use))]` is set.
///   - `ptx::local_memory_use`: check for any usage of local memory, which may
///     slow down kernel execution. By default,
///     `#[kernel(warn(ptx::local_memory_use))]` is set.
///   - `ptx::register_spills`: check for any spills of registers to local
///     memory. While using less registers can allow more kernels to be run in
///     parallel, register spills may also point to missed optimisations. By
///     default, `#[kernel(warn(ptx::register_spills))]` is set.
///   - `ptx::dynamic_stack_size`: check if the PTX compiler is unable to
///     statically determine the size of the required kernel function stack.
///     When the static stack size is known, the compiler may be able to keep it
///     entirely within the fast register file. However, when the stack size is
///     dynamic, more costly memory load and store operations are needed. By
///     default, `#[kernel(warn(ptx::dynamic_stack_size))]` is set.
///   - `ptx::verbose`: utility lint to output verbose PTX compiler messages as
///     warnings (`warn`) or errors (`deny` or `forbid`) or to not output them
///     (`allow`). By default, `#[kernel(allow(ptx::verbose))]` is set.
///   - `ptx::dump_assembly`: utility lint to output the compiled PTX assembly
///     code as a warning (`warn`) or an error (`deny` or `forbid`) or to not
///     output it (`allow`). By default, `#[kernel(allow(ptx::dump_assembly))]`
///     is set.
///
/// [`rust_cuda::kernel::TypedPtxKernel`]: https://juntyr.github.io/rust-cuda/rust_cuda/kernel/struct.TypedPtxKernel.html
/// [`rust_cuda::kernel::CudaKernelParameter`]: https://juntyr.github.io/rust-cuda/rust_cuda/kernel/trait.CudaKernelParameter.html
/// [`rust_cuda::kernel::CompiledKernelPtx`]: https://juntyr.github.io/rust-cuda/rust_cuda/kernel/trait.CompiledKernelPtx.html
/// [`rust-cuda`]: https://juntyr.github.io/rust-cuda/rust_cuda
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    kernel::wrapper::kernel(attr, func)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
/// Helper macro to specialise the generic kernel param types when compiling
/// the specialised kernel for CUDA.
pub fn specialise_kernel_param_type(tokens: TokenStream) -> TokenStream {
    kernel::specialise::param_type::specialise_kernel_param_type(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
/// Helper macro to specialise the CUDA kernel entry point name, used on the
/// host for linking to it.
pub fn specialise_kernel_entry_point(tokens: TokenStream) -> TokenStream {
    kernel::specialise::entry_point::specialise_kernel_entry_point(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro_attribute]
/// Helper macro to specialise the name of the CUDA kernel function item, used
/// to give each specialised version a unique ident when compiling for CUDA.
pub fn specialise_kernel_function(attr: TokenStream, func: TokenStream) -> TokenStream {
    kernel::specialise::function::specialise_kernel_function(attr, func)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
/// Helper macro to cheaply check the generic CUDA kernel, used on the host to
/// provide code error feedback even when no specialised kernel is linked.
pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    kernel::link::check_kernel(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
/// Helper macro to compile a specialised CUDA kernel and produce its PTX
/// assembly code, which is used on the host when linking specialised kernels.
pub fn compile_kernel(tokens: TokenStream) -> TokenStream {
    kernel::link::compile_kernel(tokens)
}
