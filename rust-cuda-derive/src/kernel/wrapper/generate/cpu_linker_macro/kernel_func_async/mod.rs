use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, KernelConfig};

mod async_func_types;
mod launch_types;
mod type_wrap;

use async_func_types::generate_async_func_types;
pub(super) use launch_types::generate_launch_types;
use type_wrap::generate_func_input_and_ptx_jit_wraps;

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_kernel_func_async(
    crate_path: &syn::Path,
    config @ KernelConfig { args, .. }: &KernelConfig,
    decl_generics @ DeclGenerics {
        generic_wrapper_params,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    FuncIdent {
        func_ident_async, ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let new_func_inputs_async = generate_async_func_types(
        crate_path,
        config,
        decl_generics,
        func_inputs,
        macro_type_ids,
    );
    let (func_input_wrap, func_cpu_ptx_jit_wrap) =
        generate_func_input_and_ptx_jit_wraps(crate_path, func_inputs);
    let (cpu_func_types_launch, cpu_func_lifetime_erased_types, cpu_func_unboxed_types) =
        generate_launch_types(
            crate_path,
            config,
            decl_generics,
            func_inputs,
            macro_type_ids,
        );

    quote! {
        #(#func_attrs)*
        #[allow(clippy::extra_unused_type_parameters)]
        fn #func_ident_async <'stream, #generic_wrapper_params>(
            &mut self,
            stream: &'stream #crate_path::rustacuda::stream::Stream,
            #(#new_func_inputs_async),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>
            #generic_wrapper_where_clause
        {
            let #crate_path::host::LaunchPackage {
                kernel, watcher, config
            } = #crate_path::host::Launcher::get_launch_package(self);

            let kernel_jit_result = if config.ptx_jit {
                kernel.compile_with_ptx_jit_args(#func_cpu_ptx_jit_wrap)?
            } else {
                kernel.compile_with_ptx_jit_args(None)?
            };

            let function = match kernel_jit_result {
                #crate_path::host::KernelJITResult::Recompiled(function) => {
                    // Call launcher hook on kernel compilation
                    <Self as #crate_path::host::Launcher>::on_compile(function, watcher)?;

                    function
                },
                #crate_path::host::KernelJITResult::Cached(function) => function,
            };

            #[allow(clippy::redundant_closure_call)]
            (|#(#func_params: #cpu_func_types_launch),*| {
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
                    fn assert_impl_devicecopy<T: #crate_path::rustacuda_core::DeviceCopy>(_val: &T) {}

                    #[allow(dead_code)]
                    fn assert_impl_no_aliasing<T: #crate_path::safety::NoAliasing>() {}

                    #[allow(dead_code)]
                    fn assert_impl_fits_into_device_register<
                        T: #crate_path::safety::FitsIntoDeviceRegister,
                    >(_val: &T) {}

                    #(assert_impl_devicecopy(&#func_params);)*
                    #(assert_impl_no_aliasing::<#cpu_func_unboxed_types>();)*
                    #(assert_impl_fits_into_device_register(&#func_params);)*
                }

                let #crate_path::host::LaunchConfig {
                    grid, block, shared_memory_size, ptx_jit: _,
                } = config;

                unsafe { stream.launch(function, grid, block, shared_memory_size,
                    &[
                        #(
                            &#func_params as *const _ as *mut ::std::ffi::c_void
                        ),*
                    ]
                ) }
            })(#(#func_input_wrap),*)
        }
    }
}
