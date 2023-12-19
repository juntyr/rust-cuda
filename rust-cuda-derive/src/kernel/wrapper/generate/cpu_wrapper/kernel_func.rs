use proc_macro2::TokenStream;

use super::super::super::{
    DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, InputCudaType, KernelConfig,
};

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)] // FIXME
pub(super) fn quote_kernel_func_inputs(
    crate_path: &syn::Path,
    KernelConfig { kernel, args, visibility, .. }: &KernelConfig,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    DeclGenerics {
        generic_kernel_params,
        generic_wrapper_params,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    inputs @ FunctionInputs { func_inputs, .. }: &FunctionInputs,
    fn_ident @ FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let launcher_predicate = quote! {
        Self: Sized + #crate_path::host::Launcher<
            KernelTraitObject = dyn #kernel #ty_generics
        >
    };

    let generic_wrapper_where_clause = match generic_wrapper_where_clause {
        Some(syn::WhereClause {
            where_token,
            predicates,
        }) if !predicates.is_empty() => {
            let comma = if predicates.empty_or_trailing() {
                quote!()
            } else {
                quote!(,)
            };

            quote! {
                #where_token #predicates #comma #launcher_predicate
            }
        },
        _ => quote! {
            where #launcher_predicate
        },
    };

    let (kernel_func_inputs, kernel_func_input_tys): (Vec<_>, Vec<_>) = func_inputs
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
                let syn_type: syn::Type = syn::parse_quote! {
                    <() as #args #ty_generics>::#type_ident
                };
                let syn_type = if let syn::Type::Reference(syn::TypeReference {
                    and_token,
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    syn::Type::Reference(syn::TypeReference {
                        and_token: *and_token,
                        lifetime: lifetime.clone(),
                        mutability: *mutability,
                        elem: Box::new(syn_type),
                    })
                } else {
                    syn_type
                };

                let param = quote! {
                    #(#attrs)* #pat #colon_token #syn_type
                };

                (param, syn_type)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip();

    let raw_func_input_wrap =
        generate_raw_func_input_wrap(crate_path, inputs, fn_ident, func_params);

    let full_generics = generic_kernel_params.iter().map(|param| match param {
        syn::GenericParam::Type(syn::TypeParam { ident, .. }) | syn::GenericParam::Const(syn::ConstParam { ident, .. }) => quote!(#ident),
        syn::GenericParam::Lifetime(syn::LifetimeDef { lifetime, .. }) => quote!(#lifetime),
    }).collect::<Vec<_>>();
    
    let ty_turbofish = ty_generics.as_turbofish();

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(non_camel_case_types)]
        #visibility type #func_ident <#generic_kernel_params> = impl Copy + Fn(
            &mut #crate_path::host::Launcher<#func_ident <#(#full_generics),*>>,
            #(#kernel_func_input_tys),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>;

        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::needless_lifetimes)]
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        #[allow(unused_variables)]
        #visibility fn #func_ident </*'stream,*/ #generic_kernel_params>(
            // &mut self,
            // TODO: move the stream
            // stream: &'stream #crate_path::rustacuda::stream::Stream,
            // kernel: &mut #crate_path::host::TypedKernel<#func_ident #ty_generics>,
            launcher: &mut #crate_path::host::Launcher<#func_ident #ty_generics>,
            #(#kernel_func_inputs),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>
        // TODO: don't allow where clause
            //#generic_wrapper_where_clause
        {
            let _: #func_ident <#(#full_generics),*> = #func_ident #ty_turbofish;

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

            todo!()

            // #raw_func_input_wrap
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
                },
                syn::FnArg::Receiver(_) => unreachable!(),
            },
        )
}
