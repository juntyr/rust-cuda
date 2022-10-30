use proc_macro2::TokenStream;
use quote::quote;

pub fn cuda_struct_declaration(
    struct_attrs_cuda: &[syn::Attribute],
    struct_layout_attrs: &[syn::Attribute],
    struct_vis_cuda: &syn::Visibility,
    struct_name_cuda: &syn::Ident,
    struct_generics_cuda: &syn::Generics,
    struct_fields_cuda: &syn::Fields,
    struct_semi_cuda: Option<syn::token::Semi>,
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = struct_generics_cuda.split_for_impl();

    let struct_repr = if struct_attrs_cuda
        .iter()
        .any(|attr| attr.path.is_ident("repr"))
    {
        quote! {}
    } else {
        quote! { #[repr(C)] }
    };

    let struct_fields_where_clause = if let Some(struct_semi_cuda) = struct_semi_cuda {
        quote!(#struct_fields_cuda #where_clause #struct_semi_cuda)
    } else {
        quote!(#where_clause #struct_fields_cuda)
    };

    quote! {
        #[allow(dead_code)]
        #[doc(hidden)]
        #(#struct_attrs_cuda)*
        #[derive(rust_cuda::const_type_layout::TypeLayout)]
        #struct_repr
        #(#struct_layout_attrs)*
        #struct_vis_cuda struct #struct_name_cuda #struct_generics_cuda #struct_fields_where_clause

        // #[derive(DeviceCopy)] can interfer with type parameters
        unsafe impl #impl_generics rust_cuda::rustacuda_core::DeviceCopy
            for #struct_name_cuda #ty_generics #where_clause {}
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rust_to_cuda_trait(
    struct_name: &syn::Ident,
    struct_name_cuda: &syn::Ident,
    struct_generics_cuda: &syn::Generics,
    struct_fields_cuda: &syn::Fields,
    combined_cuda_alloc_type: &TokenStream,
    r2c_field_declarations: &[TokenStream],
    r2c_field_initialisations: &[TokenStream],
    r2c_field_destructors: &[TokenStream],
) -> TokenStream {
    let rust_to_cuda_struct_construction = match struct_fields_cuda {
        syn::Fields::Named(_) => quote! {
            #struct_name_cuda {
                #(#r2c_field_initialisations)*
            }
        },
        syn::Fields::Unnamed(_) => quote! {
            #struct_name_cuda (
                #(#r2c_field_initialisations)*
            )
        },
        syn::Fields::Unit => quote! { #struct_name_cuda },
    };

    let (impl_generics, ty_generics, where_clause) = struct_generics_cuda.split_for_impl();

    quote! {
        unsafe impl #impl_generics rust_cuda::common::RustToCuda for #struct_name #ty_generics
            #where_clause
        {
            type CudaRepresentation = #struct_name_cuda #ty_generics;

            #[cfg(not(target_os = "cuda"))]
            type CudaAllocation = #combined_cuda_alloc_type;

            #[cfg(not(target_os = "cuda"))]
            unsafe fn borrow<CudaAllocType: rust_cuda::host::CudaAlloc>(
                &self, alloc: CudaAllocType
            ) -> rust_cuda::rustacuda::error::CudaResult<(
                rust_cuda::common::DeviceAccessible<Self::CudaRepresentation>,
                rust_cuda::host::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>
            )> {
                let alloc_front = rust_cuda::host::NullCudaAlloc;
                let alloc_tail = alloc;

                #(#r2c_field_declarations)*

                let borrow = #rust_to_cuda_struct_construction;

                Ok((
                    rust_cuda::common::DeviceAccessible::from(borrow),
                    rust_cuda::host::CombinedCudaAlloc::new(alloc_front, alloc_tail)
                ))
            }

            #[cfg(not(target_os = "cuda"))]
            unsafe fn restore<CudaAllocType: rust_cuda::host::CudaAlloc>(
                &mut self,
                alloc: rust_cuda::host::CombinedCudaAlloc<
                    Self::CudaAllocation, CudaAllocType
                >,
            ) -> rust_cuda::rustacuda::error::CudaResult<CudaAllocType> {
                let (alloc_front, alloc_tail) = alloc.split();

                #(#r2c_field_destructors)*

                Ok(alloc_tail)
            }
        }
    }
}

pub fn cuda_as_rust_trait(
    struct_name: &syn::Ident,
    struct_name_cuda: &syn::Ident,
    struct_generics_cuda: &syn::Generics,
    struct_fields_cuda: &syn::Fields,
    c2r_field_initialisations: &[TokenStream],
) -> TokenStream {
    let cuda_as_rust_struct_construction = match struct_fields_cuda {
        syn::Fields::Named(_) => quote! {
            #struct_name {
                #(#c2r_field_initialisations)*
            }
        },
        syn::Fields::Unnamed(_) => quote! {
            #struct_name (
                #(#c2r_field_initialisations)*
            )
        },
        syn::Fields::Unit => quote! { #struct_name },
    };

    let (impl_generics, ty_generics, where_clause) = &struct_generics_cuda.split_for_impl();

    quote! {
        unsafe impl #impl_generics rust_cuda::common::CudaAsRust
            for #struct_name_cuda #ty_generics #where_clause
        {
            type RustRepresentation = #struct_name #ty_generics;

            #[cfg(target_os = "cuda")]
            unsafe fn as_rust(
                this: &rust_cuda::common::DeviceAccessible<Self>,
            ) -> #struct_name #ty_generics {
                #cuda_as_rust_struct_construction
            }
        }
    }
}
