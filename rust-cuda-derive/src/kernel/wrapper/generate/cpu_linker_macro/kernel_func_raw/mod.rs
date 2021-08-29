use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, KernelConfig};

mod launch_types;
mod raw_func_types;
mod type_wrap;

use launch_types::generate_launch_types;
use raw_func_types::generate_raw_func_types;
use type_wrap::generate_func_input_and_ptx_jit_wraps;

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_kernel_func_raw(
    config @ KernelConfig { args, .. }: &KernelConfig,
    decl_generics
    @
    DeclGenerics {
        generic_start_token,
        generic_wrapper_params,
        generic_close_token,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    FuncIdent { func_ident_raw, .. }: &FuncIdent,
    func_params: &[syn::Pat],
    func_attrs: &[syn::Attribute],
    func_type_errors: &[syn::Ident],
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let arch_checks = super::super::arch_checks::quote_arch_checks();
    let new_func_inputs_raw =
        generate_raw_func_types(config, decl_generics, func_inputs, macro_type_ids);
    let (func_input_wrap, func_cpu_ptx_jit_wrap) =
        generate_func_input_and_ptx_jit_wraps(func_inputs);
    let (cpu_func_types_launch, cpu_func_lifetime_erased_types, cpu_func_unboxed_types) =
        generate_launch_types(config, decl_generics, func_inputs, macro_type_ids);

    quote! {
        #(#func_attrs)*
        fn #func_ident_raw #generic_start_token #generic_wrapper_params #generic_close_token (
            &mut self, #(#new_func_inputs_raw),*
        ) -> rust_cuda::rustacuda::error::CudaResult<()>
            #generic_wrapper_where_clause
        {
            use rust_cuda::ptx_jit::host::compiler::PtxJITResult;

            #arch_checks

            #[repr(C)]
            struct TypedKernel {
                compiler: rust_cuda::ptx_jit::host::compiler::PtxJITCompiler,
                kernel: Option<rust_cuda::ptx_jit::host::kernel::CudaKernel>,
                entry_point: Box<[u8]>,
            }

            let kernel = rust_cuda::host::Launcher::get_kernel_mut(self);
            let typed_kernel: &mut TypedKernel = unsafe { &mut *(
                kernel as *mut rust_cuda::host::TypedKernel<_> as *mut TypedKernel
            ) };
            let compiler = &mut typed_kernel.compiler;

            let function = match (rust_cuda::ptx_jit::compilePtxJITwithArguments! {
                compiler(#(#func_cpu_ptx_jit_wrap),*)
            }, typed_kernel.kernel.as_mut()) {
                (
                    PtxJITResult::Cached(_),
                    Some(kernel),
                ) => kernel,
                (
                    PtxJITResult::Cached(ptx_cstr)
                    | PtxJITResult::Recomputed(ptx_cstr),
                    _,
                ) => {
                    // Safety: `entry_point` is created using
                    //         `CString::into_bytes_with_nul`
                    let entry_point_cstr = unsafe {
                        ::std::ffi::CStr::from_bytes_with_nul_unchecked(
                            &typed_kernel.entry_point
                        )
                    };

                    let kernel = rust_cuda::ptx_jit::host::kernel::CudaKernel::new(
                        ptx_cstr, entry_point_cstr
                    )?;

                    // Call launcher hook on kernel compilation
                    rust_cuda::host::Launcher::on_compile(
                        self, kernel.get_function()
                    )?;

                    // Replace the existing compiled kernel, drop the old one
                    typed_kernel.kernel.insert(kernel)
                },
            }.get_function();

            #[allow(clippy::redundant_closure_call)]
            (|#(#func_params: #cpu_func_types_launch),*| {
                #(
                    #[allow(dead_code, non_camel_case_types)]
                    enum #func_type_errors {}
                    const _: [#func_type_errors; 1 - {
                        const ASSERT: bool = (
                            ::std::mem::size_of::<#cpu_func_lifetime_erased_types>() <= 8
                        ); ASSERT
                    } as usize] = [];
                )*

                #[deny(improper_ctypes)]
                mod __rust_cuda_ffi_safe_assert {
                    use super::#args;

                    extern "C" { #(
                        #[allow(dead_code)]
                        static #func_params: #cpu_func_lifetime_erased_types;
                    )* }
                }

                if false {
                    #[allow(dead_code)]
                    fn assert_impl_devicecopy<
                        T: rust_cuda::rustacuda_core::DeviceCopy
                    >(_val: &T) {}

                    #[allow(dead_code)]
                    fn assert_impl_no_aliasing<
                        T: rust_cuda::utils::aliasing::NoAliasing
                    >() {}

                    #(assert_impl_devicecopy(&#func_params);)*
                    #(assert_impl_no_aliasing::<#cpu_func_unboxed_types>();)*
                }

                let stream = rust_cuda::host::Launcher::get_stream(self);
                let rust_cuda::host::LaunchConfig {
                    grid, block, shared_memory_size
                } = rust_cuda::host::Launcher::get_config(self);

                unsafe { stream.launch(function, grid, block, shared_memory_size,
                    &[
                        #(
                            &#func_params as *const _ as *mut ::std::ffi::c_void
                        ),*
                    ]
                ) }?;

                stream.synchronize()
            })(#(#func_input_wrap),*)
        }
    }
}
