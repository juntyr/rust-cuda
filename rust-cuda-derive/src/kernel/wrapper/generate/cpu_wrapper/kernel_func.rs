use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, InputCudaType};

pub(super) fn quote_kernel_func_inputs(
    crate_path: &syn::Path,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    DeclGenerics {
        generic_kernel_params,
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    inputs @ FunctionInputs { func_inputs, .. }: &FunctionInputs,
    fn_ident @ FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let kernel_func_inputs = func_inputs.iter().collect::<Vec<_>>();
    let kernel_func_input_tys = func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => syn::Type::clone(ty),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let launcher = syn::Ident::new("launcher", proc_macro2::Span::mixed_site());

    let raw_func_input_wrap =
        generate_raw_func_input_wrap(crate_path, inputs, fn_ident, func_params, &launcher);

    let full_generics = generic_kernel_params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(syn::TypeParam { ident, .. })
            | syn::GenericParam::Const(syn::ConstParam { ident, .. }) => quote!(#ident),
            syn::GenericParam::Lifetime(syn::LifetimeDef { lifetime, .. }) => quote!(#lifetime),
        })
        .collect::<Vec<_>>();

    let ty_turbofish = ty_generics.as_turbofish();

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(non_camel_case_types)]
        pub type #func_ident #generic_start_token
            #generic_kernel_params
        #generic_close_token = impl Copy + Fn(
            &mut #crate_path::host::Launcher<#func_ident #generic_start_token
                #(#full_generics),*
            #generic_close_token>,
            #(#kernel_func_input_tys),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>;

        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::needless_lifetimes)]
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        #[allow(unused_variables)]
        pub fn #func_ident <#generic_kernel_params>(
            #launcher: &mut #crate_path::host::Launcher<#func_ident #ty_generics>,
            #(#kernel_func_inputs),*
        ) -> #crate_path::rustacuda::error::CudaResult<()> {
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
    launcher: &syn::Ident,
) -> TokenStream {
    func_inputs
        .iter()
        .zip(func_params)
        .zip(func_input_cuda_types.iter())
        .rev()
        .fold(
            quote! {
                #func_ident_async(#launcher, #(#func_params),*)?;
                #launcher.stream.synchronize()
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
