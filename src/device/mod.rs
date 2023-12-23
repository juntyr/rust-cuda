use core::mem::ManuallyDrop;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{specialise_kernel_function, specialise_kernel_type};

use crate::{
    common::{
        CudaAsRust, DeviceAccessible, DeviceConstRef, DeviceMutRef, DeviceOwnedRef, RustToCuda,
    },
    safety::{NoSafeAliasing, SafeDeviceCopy},
};

pub mod alloc;
pub mod thread;
pub mod utils;

pub trait BorrowFromRust: RustToCuda + NoSafeAliasing {
    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  [`DeviceConstRef`] borrowed on the CPU using the corresponding
    ///  [`LendToCuda::lend_to_cuda`](crate::host::LendToCuda::lend_to_cuda).
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr_mut` is the
    ///  [`DeviceMutRef`] borrowed on the CPU using the corresponding
    ///  [`LendToCuda::lend_to_cuda_mut`](crate::host::LendToCuda::lend_to_cuda_mut).
    /// Furthermore, since different GPU threads can access heap storage
    ///  mutably inside the safe `inner` scope, there must not be any
    ///  aliasing between concurrently running threads.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  [`DeviceOwnedRef`] borrowed on the CPU using the corresponding
    ///  [`LendToCuda::move_to_cuda`](crate::host::LendToCuda::move_to_cuda).
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        cuda_repr_mut: DeviceOwnedRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy;
}

impl<T: RustToCuda + NoSafeAliasing> BorrowFromRust for T {
    #[inline]
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // `rust_repr` must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let rust_repr = ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr.as_ref()));

        inner(&rust_repr)
    }

    #[inline]
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        mut cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // `rust_repr_mut` must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let mut rust_repr_mut = ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr_mut.as_mut()));

        // Ideally, we would only provide access to `Pin<&mut T>` s.t.
        // `core::mem::replace`  cannot be used.
        // However, we should still be fine because
        // - the shallow part of `rust_repr_mut` is a unique copy per thread, so
        //   replacing it affects no other thread (immediately, drop is handled below)
        // - any deep parts of `rust_repr_mut` are not allowed to hand out aliasing
        //   mutable references, so any deep memory replacement would not affect other
        //   threads
        // - since any deep data is allocated from the host, we *hope* that trying to
        //   drop it in a CUDA thread turns into a no-op
        inner(&mut rust_repr_mut)
    }

    #[inline]
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        mut cuda_repr_mut: DeviceOwnedRef<
            DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>,
        >,
        inner: F,
    ) -> O
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy,
    {
        inner(CudaAsRust::as_rust(cuda_repr_mut.as_mut()))
    }
}
