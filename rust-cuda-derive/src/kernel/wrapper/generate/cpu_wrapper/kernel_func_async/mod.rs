use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, KernelConfig};

mod async_func_types;
mod launch_types;
mod type_wrap;

use async_func_types::generate_async_func_types;
use launch_types::generate_launch_types;
use type_wrap::generate_func_input_and_ptx_jit_wraps;

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_kernel_func_async(
    crate_path: &syn::Path,
    config: &KernelConfig,
    impl_generics @ ImplGenerics { ty_generics, .. }: &ImplGenerics,
    DeclGenerics {
        generic_kernel_params,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    FuncIdent {
        func_ident,
        func_ident_async,
        ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let launcher = syn::Ident::new("launcher", proc_macro2::Span::mixed_site());
    let stream = syn::Lifetime::new("'stream", proc_macro2::Span::mixed_site());

    let kernel_func_async_inputs =
        generate_async_func_types(crate_path, config, impl_generics, func_inputs, &stream);
    let (func_input_wrap, func_cpu_ptx_jit_wrap) =
        generate_func_input_and_ptx_jit_wraps(crate_path, func_inputs);
    let (cpu_func_types_launch, cpu_func_unboxed_types) =
        generate_launch_types(crate_path, config, impl_generics, func_inputs);

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::extra_unused_type_parameters)]
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        #[allow(unused_variables)]
        pub fn #func_ident_async <#stream, #generic_kernel_params>(
            #launcher: &mut #crate_path::host::Launcher<#stream, '_, #func_ident #ty_generics>,
            #(#kernel_func_async_inputs),*
        ) -> #crate_path::rustacuda::error::CudaResult<()> {
            let kernel_jit_result = if #launcher.config.ptx_jit {
                #launcher.kernel.compile_with_ptx_jit_args(#func_cpu_ptx_jit_wrap)?
            } else {
                #launcher.kernel.compile_with_ptx_jit_args(None)?
            };
            let function = match kernel_jit_result {
                #crate_path::host::KernelJITResult::Recompiled(function)
                | #crate_path::host::KernelJITResult::Cached(function) => function,
            };

            #[allow(clippy::redundant_closure_call)]
            (|#(#func_params: #cpu_func_types_launch),*| {
                if false {
                    #[allow(dead_code)]
                    fn assert_impl_devicecopy<T: #crate_path::rustacuda_core::DeviceCopy>(_val: &T) {}

                    #[allow(dead_code)]
                    fn assert_impl_no_safe_aliasing<T: #crate_path::safety::NoSafeAliasing>() {}

                    #(assert_impl_devicecopy(&#func_params);)*
                    #(assert_impl_no_safe_aliasing::<#cpu_func_unboxed_types>();)*
                }

                let #crate_path::host::LaunchConfig {
                    grid, block, shared_memory_size, ptx_jit: _,
                } = #launcher.config.clone();

                unsafe { #launcher.stream.launch(function, grid, block, shared_memory_size,
                    &[
                        #(
                            &#func_params as *const _ as *mut ::core::ffi::c_void
                        ),*
                    ]
                ) }
            })(#(#func_input_wrap),*)
        }
    }
}
