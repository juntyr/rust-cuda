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

    (quote! {
        #[cfg(not(target_os = "cuda"))]
        unsafe impl #impl_generics rust_cuda::host::LendToCuda for #struct_name #ty_generics
            #where_clause
        {
            fn lend_to_cuda<
                O,
                LendToCudaInnerFunc: FnOnce(
                    rust_cuda::common::DeviceBoxConst<
                        <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
                    >
                ) -> rustacuda::error::CudaResult<O>,
            >(
                &self,
                inner: LendToCudaInnerFunc,
            ) -> rustacuda::error::CudaResult<O> {
                use rust_cuda::common::RustToCuda;

                let (cuda_repr, tail_alloc) = unsafe {
                    self.borrow(rust_cuda::host::NullCudaAlloc)
                }?;

                let mut device_box = rust_cuda::host::CudaDropWrapper::from(
                    rustacuda::memory::DeviceBox::new(&cuda_repr)?
                );
                let device_box_const = rust_cuda::common::DeviceBoxConst::from(&device_box);

                let alloc = rust_cuda::host::CombinedCudaAlloc::new(device_box, tail_alloc);

                let result = inner(device_box_const);

                core::mem::drop(alloc);

                result
            }

            fn lend_to_cuda_mut<
                O,
                LendToCudaInnerFunc: FnOnce(
                    rust_cuda::common::DeviceBoxMut<
                        <Self as rust_cuda::common::RustToCuda>::CudaRepresentation
                    >
                ) -> rustacuda::error::CudaResult<O>,
            >(
                &mut self,
                inner: LendToCudaInnerFunc,
            ) -> rustacuda::error::CudaResult<O> {
                use rust_cuda::common::RustToCuda;

                let (cuda_repr, alloc) = unsafe {
                    self.borrow_mut(rust_cuda::host::NullCudaAlloc)
                }?;

                let mut device_box = rust_cuda::host::CudaDropWrapper::from(
                    rustacuda::memory::DeviceBox::new(&cuda_repr)?
                );
                let device_box_mut = rust_cuda::common::DeviceBoxMut::from(&mut device_box);

                let result = inner(device_box_mut);

                core::mem::drop(device_box);

                let _: rust_cuda::host::NullCudaAlloc = unsafe {
                    self.un_borrow_mut(cuda_repr, alloc)
                }?;

                result
            }
        }

        #[cfg(target_os = "cuda")]
        unsafe impl #impl_generics rust_cuda::device::BorrowFromRust for #struct_name #ty_generics
            #where_clause
        {
            unsafe fn with_borrow_from_rust<O, LendToCudaInnerFunc: FnOnce(
                &Self
            ) -> O>(
                cuda_repr: rust_cuda::common::DeviceBoxConst<<Self as rust_cuda::common::RustToCuda>::CudaRepresentation>,
                inner: LendToCudaInnerFunc,
            ) -> O {
                use rust_cuda::common::CudaAsRust;

                // This is only safe because we do not expose mutability of `rust_repr` to the `inner` closure
                let cuda_repr_mut: &mut <Self as rust_cuda::common::RustToCuda>::CudaRepresentation = &mut *(cuda_repr.as_ref() as *const _ as *mut _);

                // rust_repr must never be dropped as we do NOT own any of the
                // heap memory it might reference
                let mut rust_repr = core::mem::ManuallyDrop::new(cuda_repr_mut.as_rust());

                inner(&rust_repr)
            }

            unsafe fn with_borrow_from_rust_mut<O, LendToCudaInnerFunc: FnOnce(
                &mut Self
            ) -> O>(
                mut cuda_repr_mut: rust_cuda::common::DeviceBoxMut<<Self as rust_cuda::common::RustToCuda>::CudaRepresentation>,
                inner: LendToCudaInnerFunc,
            ) -> O {
                use rust_cuda::common::CudaAsRust;

                // rust_repr must never be dropped as we do NOT own any of the
                // heap memory it might reference
                let mut rust_repr = core::mem::ManuallyDrop::new(cuda_repr_mut.as_mut().as_rust());

                inner(&mut rust_repr)
            }
        }
    })
    .into()
}
