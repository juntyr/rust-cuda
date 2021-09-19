use crate::{
    common::{CudaAsRust, DeviceAccessible, RustToCuda},
    memory::SafeDeviceCopy,
};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct SafeDeviceCopyWrapper<T: SafeDeviceCopy>(T);

unsafe impl<T: SafeDeviceCopy> rustacuda_core::DeviceCopy for SafeDeviceCopyWrapper<T> {}

impl<T: SafeDeviceCopy> From<T> for SafeDeviceCopyWrapper<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: SafeDeviceCopy> SafeDeviceCopyWrapper<T> {
    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn from_ref(reference: &T) -> &Self {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { &*(reference as *const T).cast() }
    }

    pub fn into_ref(&self) -> &T {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { &*(self as *const Self).cast() }
    }

    pub fn from_mut(reference: &mut T) -> &mut Self {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { &mut *(reference as *mut T).cast() }
    }

    pub fn into_mut(&mut self) -> &mut T {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { &mut *(self as *mut Self).cast() }
    }

    pub fn from_slice(slice: &[T]) -> &[Self] {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    pub fn into_slice(slice: &[Self]) -> &[T] {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    pub fn from_mut_slice(slice: &mut [T]) -> &mut [Self] {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    pub fn into_mut_slice(slice: &mut [Self]) -> &mut [T] {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }
}

unsafe impl<T: SafeDeviceCopy> RustToCuda for SafeDeviceCopyWrapper<T> {
    #[cfg(feature = "host")]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = Self;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc);
        Ok((DeviceAccessible::from(&self.0), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail): (crate::host::NullCudaAlloc, A) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: SafeDeviceCopy> CudaAsRust for SafeDeviceCopyWrapper<T> {
    type RustRepresentation = Self;

    #[cfg(any(not(feature = "host"), doc))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&**this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}
