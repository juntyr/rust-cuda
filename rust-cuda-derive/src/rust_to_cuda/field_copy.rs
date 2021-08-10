use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};

use super::CudaReprFieldTy;

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn impl_field_copy_init_and_expand_alloc_type(
    field: &syn::Field,
    field_index: usize,

    cuda_repr_field_ty: Option<CudaReprFieldTy>,

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

    if let Some(CudaReprFieldTy::Embedded(field_type)) = cuda_repr_field_ty {
        combined_cuda_alloc_type = quote! {
            rust_cuda::host::CombinedCudaAlloc<
                <#field_type as rust_cuda::common::RustToCuda>::CudaAllocation,
                #combined_cuda_alloc_type
            >
        };

        r2c_field_declarations.push(quote! {
            let (#field_repr_ident, alloc_front) = self.#field_accessor.borrow_mut(
                alloc_front
            )?;
        });

        r2c_field_initialisations.push(quote! {
            #optional_field_ident #field_repr_ident,
        });

        r2c_field_destructors.push(quote! {
            let alloc_front = self.#field_accessor.un_borrow_mut(
                cuda_repr.#field_accessor,
                alloc_front,
            )?;
        });

        c2r_field_initialisations.push(quote! {
            #optional_field_ident self.#field_accessor.as_rust(),
        });
    } else {
        r2c_field_initialisations.push(quote! {
            #optional_field_ident self.#field_accessor.clone(),
        });

        c2r_field_initialisations.push(quote! {
            #optional_field_ident self.#field_accessor.clone(),
        });
    }

    combined_cuda_alloc_type
}
