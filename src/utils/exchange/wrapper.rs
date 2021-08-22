use core::ops::{Deref, DerefMut};

use rustacuda::{error::CudaResult, memory::DeviceBox};
use rustacuda_core::{DeviceCopy, DevicePointer};

use crate::{
    common::{DeviceAccessible, RustToCuda},
    host::{
        CombinedCudaAlloc, CudaDropWrapper, EmptyCudaAlloc, HostDevicePointerConst,
        HostDevicePointerMut, NullCudaAlloc,
    },
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWithCudaWrapper<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: DevicePointerBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWithHostWrapper<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: DevicePointerBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    cuda_repr: DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NullCudaAlloc>,
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWithCudaWrapper<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        let (cuda_repr, _null_alloc) = unsafe { value.borrow(NullCudaAlloc) }?;

        let device_box = DevicePointerBox::new(DeviceBox::new(&cuda_repr)?);

        Ok(Self { value, device_box })
    }

    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn move_to_cuda(mut self) -> CudaResult<ExchangeWithHostWrapper<T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow(NullCudaAlloc) }?;

        self.device_box.with_box(|device_box| {
            rustacuda::memory::CopyDestination::copy_from(device_box, &cuda_repr)
        })?;

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
        let _null_alloc: NullCudaAlloc = unsafe { self.value.restore(self.null_alloc) }?;

        Ok(ExchangeWithCudaWrapper {
            value: self.value,
            device_box: self.device_box,
        })
    }

    pub fn as_ref(
        &self,
    ) -> HostDevicePointerConst<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        HostDevicePointerConst::new(self.device_box.as_ref(), &self.cuda_repr)
    }

    pub fn as_mut(
        &mut self,
    ) -> HostDevicePointerMut<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        HostDevicePointerMut::new(self.device_box.as_mut(), &mut self.cuda_repr)
    }
}

#[repr(transparent)]
struct DevicePointerBox<T: DeviceCopy>(DevicePointer<T>);

impl<T: DeviceCopy> DevicePointerBox<T> {
    pub fn new(device_box: DeviceBox<T>) -> Self {
        Self(DeviceBox::into_device(device_box))
    }

    pub fn with_box<Q, F: FnOnce(&mut DeviceBox<T>) -> Q>(&mut self, inner: F) -> Q {
        // Safety: The `DeviceBox` is recrated from the pointer coming from
        //         `DeviceBox::into_device`
        let mut device_box = unsafe { DeviceBox::from_device(self.0) };

        let result = inner(&mut device_box);

        core::mem::forget(device_box);

        result
    }

    pub fn as_ref(&self) -> &DevicePointer<T> {
        &self.0
    }

    pub fn as_mut(&mut self) -> &mut DevicePointer<T> {
        &mut self.0
    }
}

impl<T: DeviceCopy> Drop for DevicePointerBox<T> {
    fn drop(&mut self) {
        // Safety: The `DeviceBox` is recrated from the pointer coming from
        //         `DeviceBox::into_device`
        let device_box = unsafe { DeviceBox::from_device(self.0) };

        core::mem::drop(CudaDropWrapper::from(device_box));
    }
}
