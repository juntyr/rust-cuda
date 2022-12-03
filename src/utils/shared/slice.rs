#[cfg(not(target_os = "cuda"))]
use core::marker::PhantomData;

use const_type_layout::TypeGraphLayout;
use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

#[cfg(not(target_os = "cuda"))]
#[allow(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + ~const TypeGraphLayout> {
    len: usize,
    marker: PhantomData<T>,
}

#[cfg(target_os = "cuda")]
#[allow(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + ~const TypeGraphLayout> {
    shared: *mut [T],
}

#[doc(hidden)]
#[derive(TypeLayout)]
#[layout(bound = "T: 'static + ~const TypeGraphLayout")]
#[repr(C)]
pub struct ThreadBlockSharedSliceCudaRepresentation<T: 'static + ~const TypeGraphLayout> {
    len: usize,
    // Note: uses a zero-element array instead of PhantomData here so that
    //       TypeLayout can still observe T's layout
    marker: [T; 0],
}

unsafe impl<T: 'static + ~const TypeGraphLayout> DeviceCopy
    for ThreadBlockSharedSliceCudaRepresentation<T>
{
}

// #[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
// #[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
impl<T: 'static + ~const TypeGraphLayout> ThreadBlockSharedSlice<T> {
    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn new_uninit_with_len(len: usize) -> Self {
        Self {
            len,
            marker: PhantomData::<T>,
        }
    }

    #[cfg(not(target_os = "cuda"))]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(target_os = "cuda")]
    #[must_use]
    pub fn len(&self) -> usize {
        core::ptr::metadata(self.shared)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_slice_ptr(&self) -> *mut [T] {
        self.shared
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.shared.cast()
    }
}

unsafe impl<T: 'static + ~const TypeGraphLayout> RustToCuda for ThreadBlockSharedSlice<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = ThreadBlockSharedSliceCudaRepresentation<T>;

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
            DeviceAccessible::from(ThreadBlockSharedSliceCudaRepresentation {
                len: self.len,
                marker: [],
            }),
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
    for ThreadBlockSharedSliceCudaRepresentation<T>
{
    type RustRepresentation = ThreadBlockSharedSlice<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(_this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        todo!()

        // unsafe {
        //     core::arch::asm!(
        //         ".shared .align {align} .b8 rust_cuda_dynamic_shared[];",
        //         align = const(core::mem::align_of::<T>()),
        //     );
        // }

        // let base: *mut u8;

        // unsafe {
        //     core::arch::asm!(
        //         "cvta.shared.u64 {reg}, rust_cuda_dynamic_shared;",
        //         reg = out(reg64) base,
        //     );
        // }

        // let slice = core::ptr::slice_from_raw_parts_mut(
        //     base.add(self.byte_offset).cast(), self.len,
        // );
    }
}
