use core::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{specialise_kernel_entry, specialise_kernel_type};

use crate::{
    common::{CudaAsRust, DeviceAccessible, DeviceConstRef, DeviceMutRef, RustToCuda},
    utils::SafeDeviceCopy,
};

pub mod utils;

pub trait BorrowFromRust: RustToCuda {
    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  `DeviceConstRef` borrowed on the CPU using the corresponding
    ///  `LendToCuda::lend_to_cuda`.
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&ShallowCopy<Self>) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr_mut` is the
    ///  `DeviceMutRef` borrowed on the CPU using the corresponding
    ///  `LendToCuda::lend_to_cuda_mut`.
    /// Furthermore, since different GPU threads can access heap storage
    ///  mutably inside the safe `inner` scope, there must not be any
    ///  aliasing between concurrently running threads.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut ShallowCopy<Self>) -> O>(
        cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  `DeviceMutRef` borrowed on the CPU using the corresponding
    ///  `LendToCuda::move_to_cuda`.
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: Sized + SafeDeviceCopy,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy;
}

impl<T: RustToCuda> BorrowFromRust for T {
    #[inline]
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&ShallowCopy<Self>) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // rust_repr must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let rust_repr = ShallowCopy::new(CudaAsRust::as_rust(cuda_repr.as_ref()));

        inner(&rust_repr)
    }

    #[inline]
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut ShallowCopy<Self>) -> O>(
        mut cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // rust_repr must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let mut rust_repr_mut = ShallowCopy::new(CudaAsRust::as_rust(cuda_repr_mut.as_mut()));

        inner(&mut rust_repr_mut)
    }

    #[inline]
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        mut cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: Sized + SafeDeviceCopy,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy,
    {
        inner(CudaAsRust::as_rust(cuda_repr_mut.as_mut()))
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ShallowCopy<T>(ManuallyDrop<T>);

impl<T> ShallowCopy<T> {
    fn new(value: T) -> Self {
        Self(ManuallyDrop::new(value))
    }
}

impl<T> Deref for ShallowCopy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for ShallowCopy<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
