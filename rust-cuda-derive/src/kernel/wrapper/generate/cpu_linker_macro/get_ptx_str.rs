use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use crate::kernel::utils::skip_kernel_compilation;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_get_ptx_str(
    crate_path: &syn::Path,
    FuncIdent {
        func_ident,
        func_ident_hash,
        ..
    }: &FuncIdent,
    config @ KernelConfig { args, .. }: &KernelConfig,
    generics @ DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    inputs: &FunctionInputs,
    func_params: &[syn::Ident],
    macro_type_ids: &[syn::Ident],
    ptx_lint_levels: &TokenStream,
) -> TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    let cpu_func_lifetime_erased_types =
        generate_lifetime_erased_types(crate_path, config, generics, inputs, macro_type_ids);

    let matching_kernel_assert = if skip_kernel_compilation() {
        quote!()
    } else {
        quote::quote_spanned! { func_ident.span()=>
            const _: #crate_path::safety::kernel_signature::Assert<{
                #crate_path::safety::kernel_signature::CpuAndGpuKernelSignatures::Match
            }> = #crate_path::safety::kernel_signature::Assert::<{
                #crate_path::safety::kernel_signature::check(
                    PTX_STR.as_bytes(),
                    concat!(".visible .entry ", #crate_path::host::specialise_kernel_call!(
                        #func_ident_hash #generic_start_token
                            #($#macro_type_ids),*
                        #generic_close_token
                    )).as_bytes()
                )
            }>;
        }
    };

    let type_layout_asserts = if skip_kernel_compilation() {
        Vec::new()
    } else {
        cpu_func_lifetime_erased_types
            .iter()
            .zip(func_params.iter())
            .map(|(ty, param)| {
                let layout_param = syn::Ident::new(
                    &format!("__{func_ident_hash}_{param}_layout").to_uppercase(),
                    param.span(),
                );

                quote::quote_spanned! { ty.span()=>
                    const _: #crate_path::safety::type_layout::Assert<{
                        #crate_path::safety::type_layout::CpuAndGpuTypeLayouts::Match
                    }> = #crate_path::safety::type_layout::Assert::<{
                        #crate_path::safety::type_layout::check::<#ty>(#layout_param)
                    }>;
                }
            })
            .collect::<Vec<_>>()
    };

    quote! {
        fn get_ptx_str() -> &'static str {
            #crate_path::host::link_kernel!{
                #func_ident #func_ident_hash #args #crate_name #crate_manifest_dir #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token #ptx_lint_levels
            }

            #matching_kernel_assert

            #(#type_layout_asserts)*

            #[deny(improper_ctypes)]
            mod __rust_cuda_ffi_safe_assert {
                #[allow(unused_imports)]
                use super::#args;

                extern "C" { #(
                    #[allow(dead_code)]
                    static #func_params: #cpu_func_lifetime_erased_types;
                )* }
            }

            PTX_STR
        }
    }
}

fn generate_lifetime_erased_types(
    crate_path: &syn::Path,
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    macro_type_ids: &[syn::Ident],
) -> Vec<TokenStream> {
    let mut cpu_func_lifetime_erased_types = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .for_each(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    <() as #args #generic_start_token
                        #($#macro_type_ids),*
                    #generic_close_token>::#type_ident
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::SafeDeviceCopy => quote::quote_spanned! { ty.span()=>
                        #crate_path::utils::device_copy::SafeDeviceCopyWrapper<#syn_type>
                    },
                    InputCudaType::LendRustToCuda => quote::quote_spanned! { ty.span()=>
                        #crate_path::common::DeviceAccessible<
                            <#syn_type as #crate_path::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                cpu_func_lifetime_erased_types.push(
                    if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if mutability.is_some() {
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceMutRef<'static, #cuda_type>
                            }
                        } else {
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceConstRef<'static, #cuda_type>
                            }
                        }
                    } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                        quote::quote_spanned! { ty.span()=>
                            #crate_path::common::DeviceMutRef<'static, #cuda_type>
                        }
                    } else {
                        cuda_type
                    },
                );
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    cpu_func_lifetime_erased_types
}
