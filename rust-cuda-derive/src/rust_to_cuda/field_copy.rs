use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};

use super::field_ty::CudaReprFieldTy;

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn impl_field_copy_init_and_expand_alloc_type(
    crate_path: &syn::Path,
    field: &syn::Field,
    field_index: usize,

    cuda_repr_field_ty: &CudaReprFieldTy,

    mut combined_cuda_alloc_type: TokenStream,

    r2c_field_declarations: &mut Vec<TokenStream>,
    r2c_field_async_declarations: &mut Vec<TokenStream>,
    r2c_field_initialisations: &mut Vec<TokenStream>,
    r2c_field_destructors: &mut Vec<TokenStream>,
    r2c_field_async_destructors: &mut Vec<TokenStream>,

    c2r_field_initialisations: &mut Vec<TokenStream>,
) -> TokenStream {
    let field_accessor = match &field.ident {
        Some(ident) => quote! { #ident },
        None => proc_macro2::Literal::usize_unsuffixed(field_index).to_token_stream(),
    };
    let field_repr_ident = match &field.ident {
        Some(ident) => format_ident!("field_{}_repr", ident),
        None => format_ident!("field_{}_repr", field_index),
    };
    let optional_field_ident = field.ident.as_ref().map(|ident| quote! { #ident: });

    match cuda_repr_field_ty {
        CudaReprFieldTy::SafeDeviceCopy => {
            r2c_field_declarations.push(quote! {
                let #field_repr_ident = #crate_path::common::DeviceAccessible::from(
                    &self.#field_accessor,
                );
            });
            r2c_field_async_declarations.push(quote! {
                let #field_repr_ident = #crate_path::common::DeviceAccessible::from(
                    &self.#field_accessor,
                );
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::common::CudaAsRust::as_rust(&this.#field_accessor).into_inner()
                },
            });
        },
        CudaReprFieldTy::RustToCuda { field_ty } => {
            combined_cuda_alloc_type = quote! {
                #crate_path::host::CombinedCudaAlloc<
                    <#field_ty as #crate_path::common::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::common::RustToCuda::borrow(
                    &self.#field_accessor,
                    alloc_front,
                )?;
            });
            r2c_field_async_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::common::RustToCudaAsync::borrow_async(
                    &self.#field_accessor,
                    alloc_front,
                    stream,
                )?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = #crate_path::common::RustToCuda::restore(
                    &mut self.#field_accessor,
                    alloc_front,
                )?;
            });
            r2c_field_async_destructors.push(quote! {
                let alloc_front = #crate_path::common::RustToCudaAsync::restore_async(
                    &mut self.#field_accessor,
                    alloc_front,
                    stream,
                )?;
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::common::CudaAsRust::as_rust(&this.#field_accessor)
                },
            });
        },
        CudaReprFieldTy::RustToCudaProxy { proxy_ty, field_ty } => {
            combined_cuda_alloc_type = quote! {
                #crate_path::host::CombinedCudaAlloc<
                    <#proxy_ty as #crate_path::common::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::common::RustToCuda::borrow(
                    <
                        #proxy_ty as #crate_path::common::RustToCudaProxy<#field_ty>
                    >::from_ref(&self.#field_accessor),
                    alloc_front,
                )?;
            });
            r2c_field_async_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::common::RustToCudaAsync::borrow_async(
                    <
                        #proxy_ty as #crate_path::common::RustToCudaAsyncProxy<#field_ty>
                    >::from_ref(&self.#field_accessor),
                    alloc_front,
                    stream,
                )?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = #crate_path::common::RustToCuda::restore(
                    <
                        #proxy_ty as #crate_path::common::RustToCudaProxy<#field_ty>
                    >::from_mut(&mut self.#field_accessor),
                    alloc_front,
                )?;
            });
            r2c_field_async_destructors.push(quote! {
                let alloc_front = #crate_path::common::RustToCudaAsync::restore_async(
                    <
                        #proxy_ty as #crate_path::common::RustToCudaAsyncProxy<#field_ty>
                    >::from_mut(&mut self.#field_accessor),
                    alloc_front,
                    stream,
                )?;
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::common::RustToCudaProxy::<#field_ty>::into(
                        #crate_path::common::CudaAsRust::as_rust(&this.#field_accessor)
                    )
                },
            });
        },
    }

    combined_cuda_alloc_type
}
