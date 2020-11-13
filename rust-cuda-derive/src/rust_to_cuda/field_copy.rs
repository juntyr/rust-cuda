use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};

use super::CudaReprFieldTy;

pub fn impl_field_copy_init_and_expand_alloc_type(
    field: &syn::Field,
    field_index: usize,

    cuda_repr_field_ty: Option<CudaReprFieldTy>,

    mut combined_cuda_alloc_type: TokenStream,

    r2c_field_declarations: &mut Vec<TokenStream>,
    r2c_field_initialisations: &mut Vec<TokenStream>,
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
        Some(CudaReprFieldTy::BoxedSlice(slice_type)) => {
            combined_cuda_alloc_type = quote! {
                rust_cuda::host::CombinedCudaAlloc<
                    rust_cuda::host::CudaDropWrapper<
                        rustacuda::memory::DeviceBuffer<
                            #slice_type
                        >
                    >,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = {
                    let mut device_buffer = rust_cuda::host::CudaDropWrapper::from(
                        rustacuda::memory::DeviceBuffer::from_slice(
                            &self.#field_accessor
                        )?
                    );

                    (
                        (device_buffer.as_device_ptr(), device_buffer.len()),
                        rust_cuda::host::CombinedCudaAlloc::new(
                            device_buffer, alloc_front
                        )
                    )
                };
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident unsafe {
                    // This is only safe because we will NOT expose mutability
                    let raw_mut_slice_ptr: *mut #slice_type =
                        self.#field_accessor.0.as_raw() as *mut #slice_type;
                    let raw_mut_slice_len = self.#field_accessor.1;

                    let raw_slice: &mut [#slice_type] = core::slice::from_raw_parts_mut(
                        raw_mut_slice_ptr, raw_mut_slice_len
                    );

                    alloc::boxed::Box::from_raw(raw_slice)
                },
            });
        },
        Some(CudaReprFieldTy::Embedded(field_type)) => {
            combined_cuda_alloc_type = quote! {
                rust_cuda::host::CombinedCudaAlloc<
                    <#field_type as rust_cuda::common::RustToCuda>::CudaAllocation,
                    #combined_cuda_alloc_type
                >
            };

            r2c_field_declarations.push(quote! {
                let (#field_repr_ident, alloc_front) = self.#field_accessor.borrow(
                    alloc_front
                )?;
            });

            r2c_field_initialisations.push(quote! {
                #optional_field_ident #field_repr_ident,
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident self.#field_accessor.as_rust(),
            });
        },
        Some(CudaReprFieldTy::Eval(eval_token_stream)) => {
            r2c_field_initialisations.push(quote! {
                #optional_field_ident self.#field_accessor.clone(),
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident #eval_token_stream,
            });
        },
        None => {
            r2c_field_initialisations.push(quote! {
                #optional_field_ident self.#field_accessor.clone(),
            });

            c2r_field_initialisations.push(quote! {
                #optional_field_ident self.#field_accessor.clone(),
            });
        },
    }

    combined_cuda_alloc_type
}
