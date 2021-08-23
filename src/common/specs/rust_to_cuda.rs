#[cfg(feature = "host")]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use crate::{
    common::{r#impl::RustToCudaImpl, sealed::RustToCudaSealed},
    utils::stack::StackOnly,
};

#[cfg(feature = "host")]
use crate::{
    common::DeviceAccessible,
    host::{CombinedCudaAlloc, NullCudaAlloc},
};

use super::cuda_as_rust::UnsafeStackOnlyDeviceCopy;

#[marker]
pub trait StackOnlyBottom {}
impl<T: StackOnly> StackOnlyBottom for T {}

#[allow(clippy::module_name_repetitions)]
pub trait RustToCudaTop: StackOnlyBottom {}
impl<T: RustToCudaImpl> StackOnlyBottom for T {}
impl<T: RustToCudaImpl> RustToCudaTop for T {}

/// CLI poke of the static (link-time) check which ensures that
///  only `StackOnly` types use this default impl
///
/// ```rust
/// #[cfg(feature = "host")]
/// unsafe {
///     let x = Box::new(true);
///
///     rust_cuda::common::RustToCuda::borrow(
///         &x, rust_cuda::host::NullCudaAlloc
///     );
/// }  
/// ```
impl<T: StackOnlyBottom> RustToCudaSealed for T {
    #[cfg(feature = "host")]
    default type CudaAllocationSealed = NullCudaAlloc;
    default type CudaRepresentationSealed = UnsafeStackOnlyDeviceCopy<T>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    default unsafe fn borrow_sealed<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationSealed>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    )> {
        // BEGIN: Static (link-time) check to ensure only `StackOnly`
        //        types use this impl
        trait IsNullCudaAlloc {
            const IMPLS: bool = false;
        }
        impl<T: ?Sized> IsNullCudaAlloc for T {}
        struct Wrapper<T: ?Sized>(core::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<
                T: StackOnly
                    + RustToCudaSealed<
                        CudaRepresentationSealed = UnsafeStackOnlyDeviceCopy<T>,
                        CudaAllocationSealed = NullCudaAlloc,
                    >,
            > Wrapper<T>
        {
            const IMPLS: bool = true;
        }

        if !<Wrapper<Self>>::IMPLS {
            extern "C" {
                fn linker_error();
            }

            linker_error();
        }
        // END: static (link-time) check

        let cuda_repr = {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping((self as *const Self).cast(), uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        #[allow(clippy::uninit_assumed_init)]
        let null_alloc = MaybeUninit::uninit().assume_init();

        Ok((cuda_repr, CombinedCudaAlloc::new(null_alloc, alloc)))
    }

    #[cfg(feature = "host")]
    default unsafe fn restore_sealed<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    ) -> rustacuda::error::CudaResult<A> {
        Ok(alloc.split().1)
    }
}

impl<T: RustToCudaTop + RustToCudaImpl> RustToCudaSealed for T {
    #[cfg(feature = "host")]
    type CudaAllocationSealed = <T as RustToCudaImpl>::CudaAllocationImpl;
    type CudaRepresentationSealed = <T as RustToCudaImpl>::CudaRepresentationImpl;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_sealed<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationSealed>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    )> {
        RustToCudaImpl::borrow_impl(self, alloc)
    }

    #[cfg(feature = "host")]
    unsafe fn restore_sealed<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    ) -> rustacuda::error::CudaResult<A> {
        RustToCudaImpl::restore_impl(self, alloc)
    }
}
