use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};

use crate::rust_to_cuda::field_ty::CudaReprFieldTy;

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn impl_field_copy_init_and_expand_alloc_type(
    crate_path: &syn::Path,
    field: &syn::Field,
    field_index: usize,

    cuda_repr_field_ty: &CudaReprFieldTy,

    mut combined_cuda_alloc_type: TokenStream,
    mut combined_cuda_alloc_async_type: TokenStream,

    r2c_field_declarations: &mut Vec<TokenStream>,
    r2c_field_async_declarations: &mut Vec<TokenStream>,
    r2c_field_async_completions: &mut Vec<syn::Ident>,
    r2c_field_initialisations: &mut Vec<TokenStream>,
    r2c_field_destructors: &mut Vec<TokenStream>,
    r2c_field_async_destructors: &mut Vec<TokenStream>,
    r2c_field_async_completion_calls: &mut Vec<TokenStream>,

    c2r_field_initialisations: &mut Vec<TokenStream>,
) -> (TokenStream, TokenStream) {
    #[expect(clippy::option_if_let_else)]
    let field_accessor = match &field.ident {
        Some(ident) => quote! { #ident },
        None => proc_macro2::Literal::usize_unsuffixed(field_index).to_token_stream(),
    };
    #[expect(clippy::option_if_let_else)]
    let field_repr_ident = match &field.ident {
        Some(ident) => format_ident!("field_{}_repr", ident),
        None => format_ident!("field_{}_repr", field_index),
    };
    #[expect(clippy::option_if_let_else)]
    let field_completion_ident = match &field.ident {
        Some(ident) => format_ident!("field_{}_completion", ident),
        None => format_ident!("field_{}_completion", field_index),
    };
    let optional_field_ident = field.ident.as_ref().map(|ident| quote! { #ident: });

    match cuda_repr_field_ty {
        CudaReprFieldTy::SafeDeviceCopy => {
            r2c_field_declarations.push(quote! {
                let #field_repr_ident = #crate_path::utils::ffi::DeviceAccessible::from(
                    &self.#field_accessor,
                );
            });
            r2c_field_async_declarations.push(quote! {
                let #field_repr_ident = #crate_path::utils::ffi::DeviceAccessible::from(
                    &self.#field_accessor,
                );
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::lend::CudaAsRust::as_rust(&this.#field_accessor).into_inner()
                },
            });
        },
        CudaReprFieldTy::RustToCuda { field_ty } => {
            combined_cuda_alloc_type = quote! {
                #crate_path::alloc::CombinedCudaAlloc<
                    <#field_ty as #crate_path::lend::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };
            combined_cuda_alloc_async_type = quote! {
                #crate_path::alloc::CombinedCudaAlloc<
                    <#field_ty as #crate_path::lend::RustToCudaAsync>::CudaAllocationAsync,
                    #combined_cuda_alloc_async_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::lend::RustToCuda::borrow(
                    &self.#field_accessor,
                    alloc_front,
                )?;
            });
            r2c_field_async_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::lend::RustToCudaAsync::borrow_async(
                    &self.#field_accessor,
                    alloc_front,
                    stream,
                )?;
                let (#field_repr_ident, #field_completion_ident) = #field_repr_ident.unwrap_unchecked()?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = #crate_path::lend::RustToCuda::restore(
                    &mut self.#field_accessor,
                    alloc_front,
                )?;
            });
            r2c_field_async_destructors.push(quote! {
                let this_backup = unsafe {
                    ::core::mem::ManuallyDrop::new(::core::ptr::read(&this))
                };
                let (r#async, alloc_front) = #crate_path::lend::RustToCudaAsync::restore_async(
                    this.map_mut(|this| &mut this.#field_accessor),
                    alloc_front,
                    stream,
                )?;
                let (value, #field_completion_ident) = r#async.unwrap_unchecked()?;
                ::core::mem::forget(value);
                let this = ::core::mem::ManuallyDrop::into_inner(this_backup);
            });

            r2c_field_async_completion_calls.push(quote! {
                #crate_path::utils::r#async::Completion::<
                    #crate_path::deps::owning_ref::BoxRefMut<'a, CudaRestoreOwner, _>
                >::complete(
                    #field_completion_ident, &mut this.#field_accessor,
                )?;
            });

            r2c_field_async_completions.push(field_completion_ident);

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::lend::CudaAsRust::as_rust(&this.#field_accessor)
                },
            });
        },
        CudaReprFieldTy::RustToCudaProxy { proxy_ty, field_ty } => {
            combined_cuda_alloc_type = quote! {
                #crate_path::alloc::CombinedCudaAlloc<
                    <#proxy_ty as #crate_path::lend::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };
            combined_cuda_alloc_async_type = quote! {
                #crate_path::alloc::CombinedCudaAlloc<
                    <#proxy_ty as #crate_path::lend::RustToCudaAsync>::CudaAllocationAsync,
                    #combined_cuda_alloc_async_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::lend::RustToCuda::borrow(
                    <
                        #proxy_ty as #crate_path::lend::RustToCudaProxy<#field_ty>
                    >::from_ref(&self.#field_accessor),
                    alloc_front,
                )?;
            });
            r2c_field_async_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = #crate_path::lend::RustToCudaAsync::borrow_async(
                    <
                        #proxy_ty as #crate_path::lend::RustToCudaProxy<#field_ty>
                    >::from_ref(&self.#field_accessor),
                    alloc_front,
                    stream,
                )?;
                let (#field_repr_ident, #field_completion_ident) = #field_repr_ident.unwrap_unchecked()?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            r2c_field_destructors.push(quote! {
                let alloc_front = #crate_path::lend::RustToCuda::restore(
                    <
                        #proxy_ty as #crate_path::lend::RustToCudaProxy<#field_ty>
                    >::from_mut(&mut self.#field_accessor),
                    alloc_front,
                )?;
            });
            r2c_field_async_destructors.push(quote! {
                let this_backup = unsafe {
                    ::core::mem::ManuallyDrop::new(::core::ptr::read(&this))
                };
                let (r#async, alloc_front) = #crate_path::lend::RustToCudaAsync::restore_async(
                    this.map_mut(|this| <
                        #proxy_ty as #crate_path::lend::RustToCudaProxy<#field_ty>
                    >::from_mut(&mut this.#field_accessor)),
                    alloc_front,
                    stream,
                )?;
                let (value, #field_completion_ident) = r#async.unwrap_unchecked()?;
                ::core::mem::forget(value);
                let this = ::core::mem::ManuallyDrop::into_inner(this_backup);
            });

            r2c_field_async_completion_calls.push(quote! {
                #crate_path::utils::r#async::Completion::<
                    #crate_path::deps::owning_ref::BoxRefMut<'a, CudaRestoreOwner, _>
                >::complete(
                    #field_completion_ident, <
                        #proxy_ty as #crate_path::lend::RustToCudaProxy<#field_ty>
                    >::from_mut(&mut this.#field_accessor),
                )?;
            });

            r2c_field_async_completions.push(field_completion_ident);

            c2r_field_initialisations.push(quote! {
                #optional_field_ident {
                    #crate_path::lend::RustToCudaProxy::<#field_ty>::into(
                        #crate_path::lend::CudaAsRust::as_rust(&this.#field_accessor)
                    )
                },
            });
        },
    }

    (combined_cuda_alloc_type, combined_cuda_alloc_async_type)
}
