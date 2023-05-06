#[cfg(not(target_os = "cuda"))]
use core::marker::PhantomData;

use const_type_layout::TypeGraphLayout;
use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

#[cfg(not(target_os = "cuda"))]
#[derive(TypeLayout)]
#[repr(transparent)]
pub struct ThreadBlockShared<T: 'static> {
    marker: PhantomData<T>,
}

#[cfg(target_os = "cuda")]
#[derive(TypeLayout)]
#[repr(transparent)]
pub struct ThreadBlockShared<T: 'static> {
    shared: *mut T,
}

#[doc(hidden)]
#[derive(TypeLayout)]
#[repr(transparent)]
pub struct ThreadBlockSharedCudaRepresentation<T: 'static> {
    // Note: uses a zero-element array instead of PhantomData here so that
    //       TypeLayout can still observe T's layout
    marker: [T; 0],
}

unsafe impl<T: 'static> DeviceCopy for ThreadBlockSharedCudaRepresentation<T> {}

impl<T: 'static> ThreadBlockShared<T> {
    #[cfg(not(target_os = "cuda"))]
    #[must_use]
    pub fn new_uninit() -> Self {
        Self {
            marker: PhantomData::<T>,
        }
    }

    #[cfg(target_os = "cuda")]
    #[must_use]
    pub fn new_uninit() -> Self {
        let shared: *mut T;

        unsafe {
            core::arch::asm!(
                ".shared .align {align} .b8 {reg}_rust_cuda_static_shared[{size}];",
                "cvta.shared.u64 {reg}, {reg}_rust_cuda_static_shared;",
                reg = out(reg64) shared,
                align = const(core::mem::align_of::<T>()),
                size = const(core::mem::size_of::<T>()),
            );
        }

        Self { shared }
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.shared
    }
}

impl<T: 'static, const N: usize> ThreadBlockShared<[T; N]> {
    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[inline]
    #[must_use]
    pub fn index(&self, index: usize) -> *const T {
        self.index_mut(index)
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[inline]
    #[must_use]
    pub fn index_mut(&self, index: usize) -> *mut T {
        assert!(index < N);

        // Safety: Since *[T; N] is valid, *T is valid iff index < N
        unsafe { self.shared.cast::<T>().add(index) }
    }
}

unsafe impl<T: 'static + ~const TypeGraphLayout> RustToCuda for ThreadBlockShared<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = ThreadBlockSharedCudaRepresentation<T>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        Ok((
            DeviceAccessible::from(ThreadBlockSharedCudaRepresentation { marker: [] }),
            crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc),
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_null, alloc): (crate::host::NullCudaAlloc, A) = alloc.split();

        Ok(alloc)
    }
}

unsafe impl<T: 'static + ~const TypeGraphLayout> CudaAsRust
    for ThreadBlockSharedCudaRepresentation<T>
{
    type RustRepresentation = ThreadBlockShared<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(_this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        ThreadBlockShared::new_uninit()
    }
}
