use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use crate::kernel::{
    wrapper::{FuncIdent, FunctionInputs, ImplGenerics},
    KERNEL_TYPE_LAYOUT_IDENT, KERNEL_TYPE_USE_END_CANARY, KERNEL_TYPE_USE_START_CANARY,
};

#[allow(clippy::too_many_lines)]
pub(in super::super) fn quote_cuda_wrapper(
    crate_path: &syn::Path,
    inputs @ FunctionInputs { func_inputs }: &FunctionInputs,
    func @ FuncIdent {
        func_ident,
        func_ident_hash,
        ..
    }: &FuncIdent,
    impl_generics @ ImplGenerics {
        impl_generics: generics,
        ..
    }: &ImplGenerics,
    func_attrs: &[syn::Attribute],
    func_params: &[syn::Ident],
) -> TokenStream {
    let (ffi_inputs, ffi_types) =
        specialise_ffi_input_types(crate_path, inputs, func, impl_generics);

    let ffi_param_ptx_jit_wrap = func_inputs.iter().enumerate().rev().fold(
        quote! {
            #func_ident(#(#func_params),*)
        },
        |inner, (i, syn::PatType { pat, ty, .. })| {
            let specialised_ty = quote::quote_spanned! { ty.span()=>
                #crate_path::device::specialise_kernel_type!(#ty for #generics in #func_ident)
            };

            // Load the device param from its FFI representation
            // To allow some parameters to also inject PTX JIT load markers here,
            //  we pass them the param index i
            quote::quote_spanned! { ty.span()=>
                unsafe {
                    <
                        #specialised_ty as #crate_path::kernel::CudaKernelParameter
                    >::with_ffi_as_device::<_, #i>(
                        #pat, |#pat: <
                            #specialised_ty as #crate_path::kernel::CudaKernelParameter
                        >::DeviceType::<'_>| { #inner }
                    )
                }
            }
        },
    );

    let private_func_params = func_params
        .iter()
        .map(|param| {
            let mut private = syn::Ident::clone(param);
            private.set_span(proc_macro::Span::def_site().into());
            private
        })
        .collect::<Vec<_>>();

    let ffi_signature_ident = syn::Ident::new(KERNEL_TYPE_LAYOUT_IDENT, func_ident.span());
    let ffi_signature_ty = quote! { extern "C" fn(#(#ffi_types),*) };

    quote! {
        #[cfg(target_os = "cuda")]
        #[#crate_path::device::specialise_kernel_function(#func_ident)]
        #[no_mangle]
        #[allow(unused_unsafe)]
        #(#func_attrs)*
        pub unsafe extern "ptx-kernel" fn #func_ident_hash(#(#ffi_inputs),*) {
            extern "C" { #(
                #[allow(dead_code)]
                #[deny(improper_ctypes)]
                static #private_func_params: #ffi_types;
            )* }

            unsafe {
                // Initialise the dynamically-sized thread-block shared memory
                //  and the thread-local offset pointer that points to it
                #crate_path::utils::shared::init();
            }

            unsafe { ::core::arch::asm!(#KERNEL_TYPE_USE_START_CANARY); }
            #[no_mangle]
            static #ffi_signature_ident: [
                u8; #crate_path::deps::const_type_layout::serialised_type_graph_len::<#ffi_signature_ty>()
            ] = #crate_path::deps::const_type_layout::serialise_type_graph::<#ffi_signature_ty>();
            unsafe { ::core::ptr::read_volatile(&#ffi_signature_ident) };
            unsafe { ::core::arch::asm!(#KERNEL_TYPE_USE_END_CANARY); }

            #ffi_param_ptx_jit_wrap
        }
    }
}

fn specialise_ffi_input_types(
    crate_path: &syn::Path,
    FunctionInputs { func_inputs }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    ImplGenerics { impl_generics, .. }: &ImplGenerics,
) -> (Vec<syn::FnArg>, Vec<syn::Type>) {
    func_inputs
        .iter()
        .map(|syn::PatType {
            attrs,
            pat,
            colon_token,
            ty,
        }| {
            let specialised_ty = quote::quote_spanned! { ty.span()=>
                #crate_path::device::specialise_kernel_type!(#ty for #impl_generics in #func_ident)
            };

            let ffi_ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                <#specialised_ty as #crate_path::kernel::CudaKernelParameter>::FfiType<'static, 'static>
            };

            let ffi_param = syn::FnArg::Typed(syn::PatType {
                attrs: attrs.clone(),
                ty: Box::new(ffi_ty.clone()),
                pat: pat.clone(),
                colon_token: *colon_token,
            });

            (ffi_param, ffi_ty)
        })
        .unzip()
}
