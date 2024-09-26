use r#final::Final;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    utils::ffi::DeviceAccessible,
};

#[doc(hidden)]
#[expect(clippy::module_name_repetitions)]
#[derive(const_type_layout::TypeLayout)]
#[repr(transparent)]
pub struct FinalCudaRepresentation<T: CudaAsRust>(DeviceAccessible<T>);

unsafe impl<T: RustToCuda> RustToCuda for Final<T> {
    type CudaAllocation = T::CudaAllocation;
    type CudaRepresentation = FinalCudaRepresentation<T::CudaRepresentation>;

    #[cfg(feature = "host")]
    unsafe fn borrow<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
    ) -> cust::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow(alloc)?;

        Ok((
            DeviceAccessible::from(FinalCudaRepresentation(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> cust::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();
        Ok(alloc_tail)
    }
}

unsafe impl<T: RustToCudaAsync> RustToCudaAsync for Final<T> {
    type CudaAllocationAsync = T::CudaAllocationAsync;

    #[cfg(feature = "host")]
    unsafe fn borrow_async<'stream, A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        crate::utils::r#async::Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow_async(alloc, stream)?;
        let (cuda_repr, completion) = unsafe { cuda_repr.unwrap_unchecked()? };

        let final_cuda_repr = DeviceAccessible::from(FinalCudaRepresentation(cuda_repr));

        let r#async = if matches!(completion, Some(crate::utils::r#async::NoCompletion)) {
            crate::utils::r#async::Async::pending(
                final_cuda_repr,
                stream,
                crate::utils::r#async::NoCompletion,
            )?
        } else {
            crate::utils::r#async::Async::ready(final_cuda_repr, stream)
        };

        Ok((r#async, alloc))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: crate::alloc::CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        crate::utils::r#async::Async<
            'a,
            'stream,
            owning_ref::BoxRefMut<'a, O, Self>,
            crate::utils::r#async::CompletionFnMut<'a, Self>,
        >,
        A,
    )> {
        let (_alloc_front, alloc_tail) = alloc.split();
        let r#async = crate::utils::r#async::Async::ready(this, stream);
        Ok((r#async, alloc_tail))
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for FinalCudaRepresentation<T> {
    type RustRepresentation = Final<T::RustRepresentation>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        Final::new(CudaAsRust::as_rust(&(**this).0))
    }
}
