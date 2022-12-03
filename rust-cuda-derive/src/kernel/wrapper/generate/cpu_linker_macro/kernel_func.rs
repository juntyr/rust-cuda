use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_kernel_func(
    crate_path: &syn::Path,
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_wrapper_params,
        generic_close_token,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    inputs @ FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    fn_ident @ FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let new_func_inputs = func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_type, _)))| match arg {
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
                    ..
                }) = &**ty
                {
                    quote! {
                        #(#attrs)* #pat #colon_token #and_token #lifetime #mutability #syn_type
                    }
                } else if matches!(cuda_type, InputCudaType::ThreadBlockShared) {
                    if let syn::Type::Slice(_) = &**ty {
                        quote! { #(#attrs)* #pat #colon_token
                            #crate_path::utils::shared::slice::ThreadBlockSharedSlice<#syn_type>
                        }
                    } else {
                        quote! { #(#attrs)* #pat #colon_token
                            #crate_path::utils::shared::r#static::ThreadBlockShared<#syn_type>
                        }
                    }
                } else {
                    quote! { #(#attrs)* #pat #colon_token #syn_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let raw_func_input_wrap =
        generate_raw_func_input_wrap(crate_path, inputs, fn_ident, func_params);

    quote! {
        #(#func_attrs)*
        #[allow(clippy::needless_lifetimes)]
        fn #func_ident <'stream, #generic_wrapper_params>(
            &mut self,
            stream: &'stream #crate_path::rustacuda::stream::Stream,
            #(#new_func_inputs),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>
            #generic_wrapper_where_clause
        {
            // impls check adapted from Nikolai Vazquez's `impls` crate:
            //  https://docs.rs/impls/1.0.3/src/impls/lib.rs.html#584-602
            const fn __check_is_sync<T: ?Sized>(_x: &T) -> bool {
                trait IsSyncMarker {
                    const SYNC: bool = false;
                }
                impl<T: ?Sized> IsSyncMarker for T {}
                struct CheckIs<T: ?Sized>(::core::marker::PhantomData<T>);
                #[allow(dead_code)]
                impl<T: ?Sized + Sync> CheckIs<T> {
                    const SYNC: bool = true;
                }

                <CheckIs<T>>::SYNC
            }

            #raw_func_input_wrap
        }
    }
}

#[allow(clippy::too_many_lines)]
fn generate_raw_func_input_wrap(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent {
        func_ident_async, ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
) -> TokenStream {
    func_inputs
        .iter()
        .zip(func_params)
        .zip(func_input_cuda_types.iter())
        .rev()
        .fold(
            quote! {
                self.#func_ident_async(stream, #(#func_params),*)?;
                stream.synchronize()
            },
            |inner, ((arg, param), (cuda_mode, _ptx_jit))| match arg {
                syn::FnArg::Typed(syn::PatType { pat, ty, .. }) => match cuda_mode {
                    InputCudaType::SafeDeviceCopy => {
                        if let syn::Type::Reference(..) = &**ty {
                            let pat_box = quote::format_ident!("__{}_box", param);

                            // DeviceCopy mode only supports immutable references
                            quote! {
                                let mut #pat_box = #crate_path::host::HostDeviceBox::from(
                                    #crate_path::rustacuda::memory::DeviceBox::new(
                                        #crate_path::utils::device_copy::SafeDeviceCopyWrapper::from_ref(#pat)
                                    )?
                                );
                                #[allow(clippy::redundant_closure_call)]
                                // Safety: `#pat_box` contains exactly the device copy of `#pat`
                                let __result = (|#pat| { #inner })(unsafe {
                                    #crate_path::host::HostAndDeviceConstRef::new(
                                        &#pat_box,  #crate_path::utils::device_copy::SafeDeviceCopyWrapper::from_ref(#pat)
                                    ).as_async()
                                });

                                #[allow(invalid_reference_casting)]
                                if !__check_is_sync(#pat) {
                                    // Safety:
                                    // * Since `#ty` is `!Sync`, it contains interior mutability
                                    // * Therefore, part of the 'immutable' device copy may have
                                    //    been mutated
                                    // * If all mutation was confined to interior mutability,
                                    //    then passing these changes on is safe (and expected)
                                    // * If any mutations occured outside interior mutability,
                                    //    then UB occurred, in the kernel (we're not the cause)
                                    #pat_box.copy_to(unsafe { &mut *(#pat as *const _ as *mut _) })?;
                                }

                                ::core::mem::drop(#pat_box);
                                __result
                            }
                        } else {
                            quote! { {
                                let #pat = #crate_path::utils::device_copy::SafeDeviceCopyWrapper::from(#pat);
                                #inner
                            } }
                        }
                    },
                    InputCudaType::LendRustToCuda => {
                        if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                            if mutability.is_some() {
                                quote! { #crate_path::host::LendToCuda::lend_to_cuda_mut(
                                    #pat, |mut #pat| { (|#pat| { #inner })(#pat.as_async()) }
                                ) }
                            } else {
                                quote! { #crate_path::host::LendToCuda::lend_to_cuda(
                                    #pat, |#pat| { (|#pat| { #inner })(#pat.as_async()) }
                                ) }
                            }
                        } else {
                            quote! { #crate_path::host::LendToCuda::move_to_cuda(
                                #pat, |mut #pat| { (|#pat| { #inner })(#pat.as_async()) }
                            ) }
                        }
                    },
                    InputCudaType::ThreadBlockShared => inner,
                },
                syn::FnArg::Receiver(_) => unreachable!(),
            },
        )
}
