use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};

use super::field_ty::CudaReprFieldTy;

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn impl_field_copy_init_and_expand_alloc_type(
    field: &syn::Field,
    field_index: usize,

    cuda_repr_field_ty: &CudaReprFieldTy,

    mut combined_cuda_alloc_type: TokenStream,

    r2c_field_declarations: &mut Vec<TokenStream>,
    r2c_field_initialisations: &mut Vec<TokenStream>,
    r2c_field_destructors: &mut Vec<TokenStream>,

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
        CudaReprFieldTy::StackOnly => {
            r2c_field_declarations.push(quote! {
                let #field_repr_ident = rust_cuda::common::DeviceAccessible::from(
                    &self.#field_accessor,
                );
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    rust_cuda::common::CudaAsRust::as_rust(&this.#field_accessor).into_inner()
                },
            });
        },
        CudaReprFieldTy::RustToCuda(cuda_repr_field_ty) => {
            combined_cuda_alloc_type = quote! {
                rust_cuda::host::CombinedCudaAlloc<
                    <#cuda_repr_field_ty as rust_cuda::common::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = rust_cuda::common::RustToCuda::borrow(
                    &self.#field_accessor,
                    alloc_front,
                )?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = rust_cuda::common::RustToCuda::restore(
                    &mut self.#field_accessor,
                    alloc_front,
                )?;
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    rust_cuda::common::CudaAsRust::as_rust(&this.#field_accessor)
                },
            });
        },
        CudaReprFieldTy::RustToCudaProxy(cuda_repr_field_proxy_ty) => {
            combined_cuda_alloc_type = quote! {
                rust_cuda::host::CombinedCudaAlloc<
                    <#cuda_repr_field_proxy_ty as rust_cuda::common::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = rust_cuda::common::RustToCuda::borrow(
                    rust_cuda::common::RustToCudaProxy::from(&self.#field_accessor),
                    alloc_front,
                )?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = rust_cuda::common::RustToCuda::restore(
                    rust_cuda::common::RustToCudaProxy::from_mut(&mut self.#field_accessor),
                    alloc_front,
                )?;
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    rust_cuda::common::RustToCudaProxy::into(
                        rust_cuda::common::CudaAsRust::as_rust(&this.#field_accessor)
                    )
                },
            });
        },
    }

    combined_cuda_alloc_type
}
