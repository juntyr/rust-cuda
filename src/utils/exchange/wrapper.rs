use core::ops::{Deref, DerefMut};

use rustacuda::{error::CudaResult, memory::DeviceBox};

use crate::{
    common::RustToCuda,
    host::{
        CombinedCudaAlloc, CudaDropWrapper, EmptyCudaAlloc, HostDeviceBoxConst, HostDeviceBoxMut,
        NullCudaAlloc,
    },
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWithCudaWrapper<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<DeviceBox<<T as RustToCuda>::CudaRepresentation>>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWithHostWrapper<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<DeviceBox<<T as RustToCuda>::CudaRepresentation>>,
    cuda_repr: <T as RustToCuda>::CudaRepresentation,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NullCudaAlloc>,
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWithCudaWrapper<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(mut value: T) -> CudaResult<Self> {
        let (cuda_repr, _null_alloc) = unsafe { value.borrow_mut(NullCudaAlloc) }?;

        let device_box = CudaDropWrapper::from(DeviceBox::new(&cuda_repr)?);

        Ok(Self { value, device_box })
    }

    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn move_to_cuda(mut self) -> CudaResult<ExchangeWithHostWrapper<T>> {
        use rustacuda::memory::CopyDestination;

        let (cuda_repr, null_alloc) = unsafe { self.value.borrow_mut(NullCudaAlloc) }?;

        self.device_box.copy_from(&cuda_repr)?;

        Ok(ExchangeWithHostWrapper {
            value: self.value,
            device_box: self.device_box,
            cuda_repr,
            null_alloc,
        })
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> Deref for ExchangeWithCudaWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> DerefMut for ExchangeWithCudaWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWithHostWrapper<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn move_to_host(mut self) -> CudaResult<ExchangeWithCudaWrapper<T>> {
        let _null_alloc: NullCudaAlloc =
            unsafe { self.value.un_borrow_mut(self.cuda_repr, self.null_alloc) }?;

        Ok(ExchangeWithCudaWrapper {
            value: self.value,
            device_box: self.device_box,
        })
    }

    pub fn as_ref(&self) -> HostDeviceBoxConst<<T as RustToCuda>::CudaRepresentation> {
        HostDeviceBoxConst::new(&self.device_box, &self.cuda_repr)
    }

    pub fn as_mut(&mut self) -> HostDeviceBoxMut<<T as RustToCuda>::CudaRepresentation> {
        HostDeviceBoxMut::new(&mut self.device_box, &self.cuda_repr)
    }
}
