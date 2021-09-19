use core::ops::{Deref, DerefMut};

use rustacuda::{error::CudaResult, memory::DeviceBox};

use crate::{
    common::{DeviceAccessible, RustToCuda},
    host::{
        CombinedCudaAlloc, EmptyCudaAlloc, HostAndDeviceConstRef, HostAndDeviceMutRef,
        HostDeviceBox, NullCudaAlloc,
    },
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHost<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDevice<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    cuda_repr: DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NullCudaAlloc>,
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnHost<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        let (cuda_repr, _null_alloc) = unsafe { value.borrow(NullCudaAlloc) }?;

        let device_box = DeviceBox::new(&cuda_repr)?.into();

        Ok(Self { value, device_box })
    }

    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn move_to_device(mut self) -> CudaResult<ExchangeWrapperOnDevice<T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow(NullCudaAlloc) }?;

        self.device_box.copy_from(&cuda_repr)?;

        Ok(ExchangeWrapperOnDevice {
            value: self.value,
            device_box: self.device_box,
            cuda_repr,
            null_alloc,
        })
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> Deref for ExchangeWrapperOnHost<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> DerefMut for ExchangeWrapperOnHost<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnDevice<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn move_to_host(mut self) -> CudaResult<ExchangeWrapperOnHost<T>> {
        let _null_alloc: NullCudaAlloc = unsafe { self.value.restore(self.null_alloc) }?;

        Ok(ExchangeWrapperOnHost {
            value: self.value,
            device_box: self.device_box,
        })
    }

    pub fn as_ref(
        &self,
    ) -> HostAndDeviceConstRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `cuda_repr`
        unsafe { HostAndDeviceConstRef::new(&self.device_box, &self.cuda_repr) }
    }

    pub fn as_mut(
        &mut self,
    ) -> HostAndDeviceMutRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `cuda_repr`
        unsafe { HostAndDeviceMutRef::new(&mut self.device_box, &mut self.cuda_repr) }
    }
}
