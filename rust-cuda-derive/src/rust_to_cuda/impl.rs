use proc_macro2::TokenStream;
use quote::quote;

#[allow(clippy::too_many_arguments)]
pub fn cuda_struct_declaration(
    crate_path: &syn::Path,
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

    #[allow(clippy::option_if_let_else)]
    let struct_fields_where_clause = if let Some(struct_semi_cuda) = struct_semi_cuda {
        quote!(#struct_fields_cuda #where_clause #struct_semi_cuda)
    } else {
        quote!(#where_clause #struct_fields_cuda)
    };

    let const_type_layout_crate_path = quote! { #crate_path::deps::const_type_layout }.to_string();

    quote! {
        #[allow(dead_code)]
        #[doc(hidden)]
        #(#struct_attrs_cuda)*
        #[derive(#crate_path::deps::const_type_layout::TypeLayout)]
        #struct_repr
        #(#struct_layout_attrs)*
        #[layout(crate = #const_type_layout_crate_path)]
        #struct_vis_cuda struct #struct_name_cuda #struct_generics_cuda #struct_fields_where_clause

        // #[derive(DeviceCopy)] can interfer with type parameters
        unsafe impl #impl_generics #crate_path::deps::rustacuda_core::DeviceCopy
            for #struct_name_cuda #ty_generics #where_clause {}
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rust_to_cuda_trait(
    crate_path: &syn::Path,
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
        unsafe impl #impl_generics #crate_path::lend::RustToCuda for #struct_name #ty_generics
            #where_clause
        {
            type CudaRepresentation = #struct_name_cuda #ty_generics;

            type CudaAllocation = #combined_cuda_alloc_type;

            #[cfg(not(target_os = "cuda"))]
            unsafe fn borrow<CudaAllocType: #crate_path::alloc::CudaAlloc>(
                &self,
                alloc: CudaAllocType,
            ) -> #crate_path::deps::rustacuda::error::CudaResult<(
                #crate_path::utils::ffi::DeviceAccessible<Self::CudaRepresentation>,
                #crate_path::alloc::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>
            )> {
                let alloc_front = #crate_path::alloc::NoCudaAlloc;
                let alloc_tail = alloc;

                #(#r2c_field_declarations)*

                let borrow = #rust_to_cuda_struct_construction;

                Ok((
                    #crate_path::utils::ffi::DeviceAccessible::from(borrow),
                    #crate_path::alloc::CombinedCudaAlloc::new(alloc_front, alloc_tail)
                ))
            }

            #[cfg(not(target_os = "cuda"))]
            unsafe fn restore<CudaAllocType: #crate_path::alloc::CudaAlloc>(
                &mut self,
                alloc: #crate_path::alloc::CombinedCudaAlloc<
                    Self::CudaAllocation, CudaAllocType
                >,
            ) -> #crate_path::deps::rustacuda::error::CudaResult<CudaAllocType> {
                let (alloc_front, alloc_tail) = alloc.split();

                #(#r2c_field_destructors)*

                Ok(alloc_tail)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rust_to_cuda_async_trait(
    crate_path: &syn::Path,
    struct_name: &syn::Ident,
    struct_name_cuda: &syn::Ident,
    struct_generics_cuda_async: &syn::Generics,
    struct_fields_cuda: &syn::Fields,
    r2c_field_async_declarations: &[TokenStream],
    r2c_field_initialisations: &[TokenStream],
    r2c_field_async_destructors: &[TokenStream],
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

    let (impl_generics, ty_generics, where_clause) = struct_generics_cuda_async.split_for_impl();

    quote! {
        unsafe impl #impl_generics #crate_path::lend::RustToCudaAsync for #struct_name #ty_generics
            #where_clause
        {
            #[cfg(not(target_os = "cuda"))]
            unsafe fn borrow_async<CudaAllocType: #crate_path::alloc::CudaAlloc>(
                &self,
                alloc: CudaAllocType,
                stream: &#crate_path::deps::rustacuda::stream::Stream,
            ) -> #crate_path::deps::rustacuda::error::CudaResult<(
                #crate_path::utils::ffi::DeviceAccessible<Self::CudaRepresentation>,
                #crate_path::alloc::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>
            )> {
                let alloc_front = #crate_path::alloc::NoCudaAlloc;
                let alloc_tail = alloc;

                #(#r2c_field_async_declarations)*

                let borrow = #rust_to_cuda_struct_construction;

                Ok((
                    #crate_path::utils::ffi::DeviceAccessible::from(borrow),
                    #crate_path::alloc::CombinedCudaAlloc::new(alloc_front, alloc_tail)
                ))
            }

            #[cfg(not(target_os = "cuda"))]
            unsafe fn restore_async<CudaAllocType: #crate_path::alloc::CudaAlloc>(
                &mut self,
                alloc: #crate_path::alloc::CombinedCudaAlloc<
                    Self::CudaAllocation, CudaAllocType
                >,
                stream: &#crate_path::deps::rustacuda::stream::Stream,
            ) -> #crate_path::deps::rustacuda::error::CudaResult<CudaAllocType> {
                let (alloc_front, alloc_tail) = alloc.split();

                #(#r2c_field_async_destructors)*

                Ok(alloc_tail)
            }
        }
    }
}

pub fn cuda_as_rust_trait(
    crate_path: &syn::Path,
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
        unsafe impl #impl_generics #crate_path::lend::CudaAsRust
            for #struct_name_cuda #ty_generics #where_clause
        {
            type RustRepresentation = #struct_name #ty_generics;

            #[cfg(target_os = "cuda")]
            unsafe fn as_rust(
                this: &#crate_path::utils::ffi::DeviceAccessible<Self>,
            ) -> #struct_name #ty_generics {
                #cuda_as_rust_struct_construction
            }
        }
    }
}
