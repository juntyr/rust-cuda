use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

pub(super) fn quote_kernel_func(
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_wrapper_params,
        generic_close_token,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    inputs @ FunctionInputs { func_inputs, .. }: &FunctionInputs,
    fn_ident @ FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
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
                    ..
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
        #[allow(clippy::needless_lifetimes)]
        fn #func_ident #generic_start_token #generic_wrapper_params #generic_close_token (
            &mut self, #(#new_func_inputs),*
        ) -> rust_cuda::rustacuda::error::CudaResult<()>
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
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent { func_ident_raw, .. }: &FuncIdent,
    func_params: &[syn::Ident],
) -> TokenStream {
    func_inputs
        .iter()
        .zip(func_params)
        .zip(func_input_cuda_types.iter())
        .rev()
        .fold(
            quote! {
                self.#func_ident_raw(#(#func_params),*)
            },
            |inner, ((arg, param), (cuda_mode, _ptx_jit))| match arg {
                syn::FnArg::Typed(syn::PatType { pat, ty, .. }) => match cuda_mode {
                    InputCudaType::SafeDeviceCopy => {
                        if let syn::Type::Reference(..) = &**ty {
                            let pat_box = quote::format_ident!("__{}_box", param);

                            // DeviceCopy mode only supports immutable references
                            quote! {
                                let mut #pat_box = rust_cuda::host::HostDeviceBox::from(
                                    rust_cuda::rustacuda::memory::DeviceBox::new(
                                        rust_cuda::utils::device_copy::SafeDeviceCopyWrapper::from_ref(#pat)
                                    )?
                                );
                                #[allow(clippy::redundant_closure_call)]
                                // Safety: `#pat_box` contains exactly the device copy of `#pat`
                                let __result = (|#pat| { #inner })(unsafe {
                                    rust_cuda::host::HostAndDeviceConstRef::new(
                                        &#pat_box,  rust_cuda::utils::device_copy::SafeDeviceCopyWrapper::from_ref(#pat)
                                    )
                                });

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
                                let #pat = rust_cuda::utils::device_copy::SafeDeviceCopyWrapper::from(#pat);
                                #inner
                            } }
                        }
                    },
                    InputCudaType::LendRustToCuda => {
                        if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                            if mutability.is_some() {
                                quote! { rust_cuda::host::LendToCuda::lend_to_cuda_mut(
                                    #pat, |#pat| { #inner }
                                ) }
                            } else {
                                quote! { rust_cuda::host::LendToCuda::lend_to_cuda(
                                    #pat, |#pat| { #inner }
                                ) }
                            }
                        } else {
                            quote! { rust_cuda::host::LendToCuda::move_to_cuda(
                                #pat, |#pat| { #inner }
                            ) }
                        }
                    },
                },
                syn::FnArg::Receiver(_) => unreachable!(),
            },
        )
}
