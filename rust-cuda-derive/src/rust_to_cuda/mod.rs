use proc_macro2::TokenStream;
use quote::{format_ident, quote};

mod field_copy;
mod field_ty;
mod generics;
mod r#impl;

use field_ty::CudaReprFieldTy;

fn get_cuda_repr_ident(rust_repr_ident: &proc_macro2::Ident) -> proc_macro2::Ident {
    format_ident!("{}CudaRepresentation", rust_repr_ident)
}

#[allow(clippy::module_name_repetitions)]
pub fn impl_rust_to_cuda(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    let (mut struct_fields_cuda, struct_semi_cuda) = if let syn::Data::Struct(s) = &ast.data {
        (s.fields.clone(), s.semi_token)
    } else {
        abort_call_site!("You can only derive the `RustToCuda` trait on structs for now.");
    };

    let struct_name = &ast.ident;
    let struct_name_cuda = get_cuda_repr_ident(struct_name);

    let mut combined_cuda_alloc_type: TokenStream = quote! {
        rust_cuda::host::NullCudaAlloc
    };
    let mut r2c_field_declarations: Vec<TokenStream> = Vec::new();
    let mut r2c_field_initialisations: Vec<TokenStream> = Vec::new();

    let mut c2r_field_initialisations: Vec<TokenStream> = Vec::new();

    match struct_fields_cuda {
        syn::Fields::Named(syn::FieldsNamed {
            named: ref mut fields,
            ..
        })
        | syn::Fields::Unnamed(syn::FieldsUnnamed {
            unnamed: ref mut fields,
            ..
        }) => {
            for (field_index, field) in fields.iter_mut().enumerate() {
                let cuda_repr_field_ty = field_ty::swap_field_type_and_get_cuda_repr_ty(field);

                combined_cuda_alloc_type = field_copy::impl_field_copy_init_and_expand_alloc_type(
                    field,
                    field_index,
                    cuda_repr_field_ty,
                    combined_cuda_alloc_type,
                    &mut r2c_field_declarations,
                    &mut r2c_field_initialisations,
                    &mut c2r_field_initialisations,
                );
            }
        }
        syn::Fields::Unit => (),
    }

    let (struct_attrs_cuda, struct_generics_cuda) =
        generics::expand_cuda_struct_generics_where_requested_in_attrs(ast);

    let cuda_struct_declaration = r#impl::cuda_struct_declaration(
        &struct_attrs_cuda,
        &ast.vis,
        &struct_name_cuda,
        &struct_generics_cuda,
        &struct_fields_cuda,
        struct_semi_cuda,
    );

    let rust_to_cuda_trait_impl = r#impl::rust_to_cuda_trait(
        &struct_name,
        &struct_name_cuda,
        &struct_generics_cuda,
        &struct_fields_cuda,
        &combined_cuda_alloc_type,
        &r2c_field_declarations,
        &r2c_field_initialisations,
    );

    let cuda_as_rust_trait_impl = r#impl::cuda_as_rust_trait(
        &struct_name,
        &struct_name_cuda,
        &struct_generics_cuda,
        &struct_fields_cuda,
        &c2r_field_initialisations,
    );

    (quote! {
        #cuda_struct_declaration

        #rust_to_cuda_trait_impl

        #cuda_as_rust_trait_impl
    })
    .into()
}
