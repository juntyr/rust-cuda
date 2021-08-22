use proc_macro::TokenStream;
use quote::quote;

use super::generics;

#[allow(clippy::module_name_repetitions)]
pub fn impl_lend_to_cuda(ast: &syn::DeriveInput) -> TokenStream {
    if !matches!(ast.data, syn::Data::Struct(_)) {
        abort_call_site!("You can only derive the `LendToCuda` trait on structs for now.");
    };

    let struct_name = &ast.ident;

    let (_r2c_attrs, r2c_generics) =
        generics::expand_cuda_struct_generics_where_requested_in_attrs(ast);
    let (impl_generics, ty_generics, where_clause) = r2c_generics.split_for_impl();

    let lend_to_cuda_impl = quote_lend_to_cuda();
    let borrow_from_rust_impl = quote_borrow_from_rust();

    (quote! {
        #[cfg(not(target_os = "cuda"))]
        unsafe impl #impl_generics rust_cuda::host::LendToCuda for #struct_name #ty_generics
            #where_clause
        {
            #lend_to_cuda_impl
        }

        #[cfg(target_os = "cuda")]
        unsafe impl #impl_generics rust_cuda::device::BorrowFromRust for #struct_name #ty_generics
            #where_clause
        {
            #borrow_from_rust_impl
        }
    })
    .into()
}

fn quote_lend_to_cuda() -> proc_macro2::TokenStream {
    quote! {
        fn lend_to_cuda<
            O,
            LendToCudaInnerFunc: FnOnce(
                rust_cuda::host::HostDevicePointerConst<
                    <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
                >
            ) -> rust_cuda::rustacuda::error::CudaResult<O>,
        >(
            &self,
            inner: LendToCudaInnerFunc,
        ) -> rust_cuda::rustacuda::error::CudaResult<O> {
            use rust_cuda::common::RustToCuda;

            let (cuda_repr, tail_alloc) = unsafe {
                self.borrow(rust_cuda::host::NullCudaAlloc)
            }?;

            let mut device_box = rust_cuda::host::CudaDropWrapper::from(
                rust_cuda::rustacuda::memory::DeviceBox::new(&cuda_repr)?
            );

            let result = inner(
                rust_cuda::host::HostDevicePointerConst::new(&device_box, &cuda_repr)
            );

            let alloc = rust_cuda::host::CombinedCudaAlloc::new(device_box, tail_alloc);

            ::core::mem::drop(alloc);

            result
        }

        fn lend_to_cuda_mut<
            O,
            LendToCudaInnerFunc: FnOnce(
                rust_cuda::host::HostDevicePointerMut<
                    <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
                >
            ) -> rust_cuda::rustacuda::error::CudaResult<O>,
        >(
            &mut self,
            inner: LendToCudaInnerFunc,
        ) -> rust_cuda::rustacuda::error::CudaResult<O> {
            use rust_cuda::common::RustToCuda;

            let (mut cuda_repr, alloc) = unsafe {
                self.borrow_mut(rust_cuda::host::NullCudaAlloc)
            }?;

            let mut device_box = rust_cuda::host::CudaDropWrapper::from(
                rust_cuda::rustacuda::memory::DeviceBox::new(&cuda_repr)?
            );

            let result = inner(
                rust_cuda::host::HostDevicePointerMut::new(&mut device_box, &mut cuda_repr)
            );

            ::core::mem::drop(device_box);

            let _: rust_cuda::host::NullCudaAlloc = unsafe {
                self.un_borrow_mut(cuda_repr, alloc)
            }?;

            result
        }
    }
}

fn quote_borrow_from_rust() -> proc_macro2::TokenStream {
    quote! {
        #[inline]
        unsafe fn with_borrow_from_rust<O, LendToCudaInnerFunc: FnOnce(
            &Self
        ) -> O>(
            cuda_repr: rust_cuda::common::DevicePointerConst<
                <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
            >,
            inner: LendToCudaInnerFunc,
        ) -> O {
            use rust_cuda::common::CudaAsRust;

            // Safety: This is only safe because we do not expose mutability of `rust_repr`
            //         to the `inner` closure
            #[allow(clippy::cast_ref_to_mut)]
            let cuda_repr_mut: &mut <
                Self as rust_cuda::common::RustToCuda
            >::CudaRepresentation = &mut *(cuda_repr.as_ref() as *const _ as *mut _);

            // rust_repr must never be dropped as we do NOT own any of the
            //  heap memory it might reference
            let mut rust_repr = ::core::mem::ManuallyDrop::new(cuda_repr_mut.as_rust());

            inner(&rust_repr)
        }

        #[inline]
        unsafe fn with_borrow_from_rust_mut<O, LendToCudaInnerFunc: FnOnce(
            &mut Self
        ) -> O>(
            mut cuda_repr_mut: rust_cuda::common::DevicePointerMut<
                <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
            >,
            inner: LendToCudaInnerFunc,
        ) -> O {
            use rust_cuda::common::CudaAsRust;

            // rust_repr must never be dropped as we do NOT own any of the
            //  heap memory it might reference
            let mut rust_repr = ::core::mem::ManuallyDrop::new(
                cuda_repr_mut.as_mut().as_rust()
            );

            inner(&mut rust_repr)
        }
    }
}
