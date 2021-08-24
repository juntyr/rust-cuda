#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{specialise_kernel_entry, specialise_kernel_type};

use crate::common::{
    CudaAsRust, DeviceAccessible, DevicePointerConst, DevicePointerMut, RustToCuda,
};

pub mod nvptx;
pub mod utils;

pub trait BorrowFromRust: RustToCuda {
    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  `DevicePointerConst` borrowed on the CPU using the corresponding
    ///  `LendToCuda::lend_to_cuda`.
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DevicePointerConst<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr_mut` is the
    ///  `DevicePointerMut` borrowed on the CPU using the corresponding
    ///  `LendToCuda::lend_to_cuda_mut`.
    /// Furthermore, since different GPU threads can access heap storage
    ///  mutably inside the safe `inner` scope, there must not be any
    ///  aliasing between concurrently running threads.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_mut: DevicePointerMut<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;
}

impl<T: RustToCuda> BorrowFromRust for T {
    #[inline]
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DevicePointerConst<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // rust_repr must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let rust_repr = core::mem::ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr.as_ref()));

        inner(&rust_repr)
    }

    #[inline]
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        mut cuda_repr_mut: DevicePointerMut<
            DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>,
        >,
        inner: F,
    ) -> O {
        // rust_repr must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let mut rust_repr_mut =
            ::core::mem::ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr_mut.as_mut()));

        inner(&mut rust_repr_mut)
    }
}
