use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

pub(super) fn quote_kernel_func(
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    inputs @ FunctionInputs { func_inputs, .. }: &FunctionInputs,
    fn_ident @ FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Pat],
    func_attrs: &[syn::Attribute],
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let new_func_inputs = func_inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! {
                    <() as #args #generic_start_token
                        #($#macro_type_ids),*
                    #generic_close_token>::#type_ident
                };

                if let syn::Type::Reference(syn::TypeReference {
                    and_token,
                    lifetime,
                    mutability,
                    elem: _elem,
                }) = &**ty
                {
                    quote! {
                        #(#attrs)* #pat #colon_token #and_token #lifetime #mutability #syn_type
                    }
                } else {
                    quote! { #(#attrs)* #pat #colon_token #syn_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let raw_func_input_wrap = generate_raw_func_input_wrap(inputs, fn_ident, func_params);

    quote! {
        #(#func_attrs)*
        fn #func_ident(&mut self, #(#new_func_inputs),*)
            -> rust_cuda::rustacuda::error::CudaResult<()>
        {
            #raw_func_input_wrap
        }
    }
}

#[allow(clippy::too_many_lines)]
fn generate_raw_func_input_wrap(
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent { func_ident_raw, .. }: &FuncIdent,
    func_params: &[syn::Pat],
) -> TokenStream {
    func_inputs
        .iter().zip(func_input_cuda_types.iter())
        .rev()
        .fold(quote! {
            self.#func_ident_raw(#(#func_params),*)
        }, |inner, (arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                match cuda_mode {
                    InputCudaType::DeviceCopy => if let syn::Type::Reference(
                        syn::TypeReference { mutability, .. }
                    ) = &**ty {
                        let pat_box = match &**pat {
                            syn::Pat::Ident(syn::PatIdent {
                                attrs,
                                by_ref: None,
                                mutability: None,
                                ident,
                                subpat: None,
                            }) => syn::Pat::Ident(syn::PatIdent {
                                attrs: attrs.clone(),
                                by_ref: None,
                                mutability: None,
                                ident: quote::format_ident!("__{}_box", ident),
                                subpat: None,
                            }),
                            _ => abort!(
                                pat.span(),
                                "Unexpected kernel input parameter: only identifiers are accepted."
                            ),
                        };

                        let pat_host_box = match &**pat {
                            syn::Pat::Ident(syn::PatIdent {
                                attrs,
                                by_ref: None,
                                mutability: None,
                                ident,
                                subpat: None,
                            }) => syn::Pat::Ident(syn::PatIdent {
                                attrs: attrs.clone(),
                                by_ref: None,
                                mutability: None,
                                ident: quote::format_ident!("__{}_host_box", ident),
                                subpat: None,
                            }),
                            _ => unreachable!(),
                        };

                        if mutability.is_some() {
                            quote! {
                                let mut #pat_box = rust_cuda::rustacuda::memory::DeviceBox::new(
                                    #pat
                                )?;
                                let mut #pat_host_box = rust_cuda::host::HostDeviceBoxMut::new(
                                    &mut #pat_box, #pat
                                );
                                let __result = {
                                    let #pat = &mut #pat_host_box;
                                    #inner
                                };
                                rust_cuda::rustacuda::memory::CopyDestination::copy_to(
                                    &#pat_box, #pat
                                )?;
                                __result
                            }
                        } else {
                            quote! {
                                let #pat_box = rust_cuda::rustacuda::memory::DeviceBox::new(#pat)?;
                                let #pat_host_box = rust_cuda::host::HostDeviceBoxConst::new(
                                    &#pat_box, #pat
                                );
                                {
                                    let #pat = &#pat_host_box;
                                    #inner
                                }
                            }
                        }
                    } else {
                        inner
                    },
                    InputCudaType::LendRustBorrowToCuda => if let syn::Type::Reference(
                        syn::TypeReference { mutability, .. }
                    ) = &**ty {
                        if mutability.is_some() {
                            quote! { rust_cuda::host::LendToCuda::lend_to_cuda_mut(
                                #pat, |mut #pat| {
                                    let #pat = &mut #pat;
                                    #inner
                                }
                            ) }
                        } else {
                            quote! { rust_cuda::host::LendToCuda::lend_to_cuda(#pat, |#pat| {
                                let #pat = &#pat;
                                #inner
                            }) }
                        }
                    } else { unreachable!() }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
}
