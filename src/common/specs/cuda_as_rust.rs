#[cfg(not(feature = "host"))]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::CudaAsRustImpl,
    sealed::{CudaAsRustSealed, RustToCudaSealed},
};

#[cfg(not(feature = "host"))]
use crate::common::DeviceAccessible;

#[repr(transparent)]
pub struct UnsafeStackOnlyDeviceCopy<T>(T);

unsafe impl<T> DeviceCopy for UnsafeStackOnlyDeviceCopy<T> {}

/// CLI poke of the static (link-time) check which ensures that
///  only `StackOnly` types use this default impl
///
/// ```rust
/// #[cfg(not(feature = "host"))]
/// unsafe {
///     let x = Box::into_raw(Box::new(true));
///     let x: rust_cuda::common::DeviceAccessible<
///         <Box::<bool> as rust_cuda::common::RustToCuda>::CudaRepresentation
///     > = std::mem::transmute(x);
///
///     rust_cuda::common::CudaAsRust::as_rust(&x);
/// }  
/// ```
impl<T: RustToCudaSealed<CudaRepresentationSealed = UnsafeStackOnlyDeviceCopy<T>>> CudaAsRustSealed
    for UnsafeStackOnlyDeviceCopy<T>
{
    type RustRepresentationSealed = T;

    #[cfg(not(feature = "host"))]
    unsafe fn as_rust_sealed(this: &DeviceAccessible<Self>) -> Self::RustRepresentationSealed {
        // BEGIN: Static (link-time) check to ensure only `StackOnly`
        //        types use this impl
        trait DoesNotImpl {
            const IMPLS: bool = false;
        }
        impl<T: ?Sized> DoesNotImpl for T {}
        struct Wrapper<T: ?Sized>(core::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<T: crate::utils::stack::StackOnly> Wrapper<UnsafeStackOnlyDeviceCopy<T>> {
            const IMPLS: bool = true;
        }

        if !<Wrapper<Self>>::IMPLS {
            extern "C" {
                fn linker_error();
            }

            linker_error();
        }
        // END: static (link-time) check

        let mut uninit = MaybeUninit::uninit();
        copy_nonoverlapping(&(**this).0, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}

impl<T: CudaAsRustImpl> CudaAsRustSealed for T {
    type RustRepresentationSealed = <Self as CudaAsRustImpl>::RustRepresentationImpl;

    #[cfg(not(feature = "host"))]
    unsafe fn as_rust_sealed(this: &DeviceAccessible<Self>) -> Self::RustRepresentationSealed {
        CudaAsRustImpl::as_rust_impl(this)
    }
}
