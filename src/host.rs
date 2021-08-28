use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
    function::Function,
    memory::{DeviceBox, DeviceBuffer, LockedBuffer},
    module::Module,
    stream::Stream,
};
use rustacuda_core::{DeviceCopy, DevicePointer};

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{link_kernel, specialise_kernel_call};

use crate::common::{DeviceAccessible, DeviceConstRef, DeviceMutRef, RustToCuda};

pub trait Launcher {
    type KernelTraitObject: ?Sized;

    fn get_config(&self) -> LaunchConfig;
    fn get_stream(&self) -> &Stream;

    fn get_kernel_mut(&mut self) -> &mut TypedKernel<Self::KernelTraitObject>;

    /// # Errors
    ///
    /// Should only return a `CudaError` if some implementation-defined
    ///  critical kernel function configuration failed.
    #[allow(unused_variables)]
    fn on_compile(&mut self, kernel: &Function) -> CudaResult<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: rustacuda::function::GridSize,
    pub block: rustacuda::function::BlockSize,
    pub shared_memory_size: u32,
}

#[repr(C)]
pub struct TypedKernel<KernelTraitObject: ?Sized> {
    _compiler: ptx_jit::host::compiler::PtxJITCompiler,
    _kernel: Option<ptx_jit::host::kernel::CudaKernel>,
    _entry_point: alloc::boxed::Box<[u8]>,
    _marker: PhantomData<KernelTraitObject>,
}

pub trait LendToCuda: RustToCuda {
    /// Lends an immutable copy of `&self` to CUDA:
    /// - code in the CUDA kernel can only access `&self` through the
    ///   `DeviceConstRef` inside the closure
    /// - after the closure, `&self` will not have changed
    ///
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda<
        O,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> CudaResult<O>,
    >(
        &self,
        inner: F,
    ) -> CudaResult<O>;

    /// Lends a mutable copy of `&mut self` to CUDA:
    /// - code in the CUDA kernel can only access `&mut self` through the
    ///   `DeviceMutRef` inside the closure
    /// - after the closure, `&mut self` might have changed in the following
    ///   ways:
    ///   - to avoid aliasing, each CUDA thread gets its own shallow copy of
    ///     `&mut self`, i.e. any shallow changes will NOT be reflected after
    ///     the closure
    ///   - each CUDA thread can access the same heap allocated storage, i.e.
    ///     any deep changes will be reflected after the closure
    ///
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda_mut<
        O,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> CudaResult<O>,
    >(
        &mut self,
        inner: F,
    ) -> CudaResult<O>;
}

impl<T: RustToCuda> LendToCuda for T {
    fn lend_to_cuda<
        O,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> CudaResult<O>,
    >(
        &self,
        inner: F,
    ) -> CudaResult<O> {
        let (cuda_repr, tail_alloc) = unsafe { self.borrow(NullCudaAlloc) }?;

        let device_box: HostDeviceBox<_> = DeviceBox::new(&cuda_repr)?.into();

        // Safety: `device_box` contains exactly the device copy of `cuda_repr`
        let result = inner(unsafe { HostAndDeviceConstRef::new(&device_box, &cuda_repr) });

        let alloc = CombinedCudaAlloc::new(device_box, tail_alloc);

        core::mem::drop(cuda_repr);
        core::mem::drop(alloc);

        result
    }

    fn lend_to_cuda_mut<
        O,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> CudaResult<O>,
    >(
        &mut self,
        inner: F,
    ) -> CudaResult<O> {
        let (mut cuda_repr, alloc) = unsafe { self.borrow(NullCudaAlloc) }?;

        let mut device_box: HostDeviceBox<_> = DeviceBox::new(&cuda_repr)?.into();

        // Safety: `device_box` contains exactly the device copy of `cuda_repr`
        let result = inner(unsafe { HostAndDeviceMutRef::new(&mut device_box, &mut cuda_repr) });

        core::mem::drop(device_box);
        core::mem::drop(cuda_repr);

        let _: NullCudaAlloc = unsafe { self.restore(alloc) }?;

        result
    }
}

pub(crate) mod private {
    pub mod alloc {
        pub trait Sealed {}
    }

    pub mod drop {
        pub trait Sealed: Sized {
            fn drop(val: Self) -> Result<(), (rustacuda::error::CudaError, Self)>;
        }
    }

    pub mod empty {
        pub trait Sealed {}
    }
}

pub trait EmptyCudaAlloc: private::empty::Sealed {}
impl<T: private::empty::Sealed> EmptyCudaAlloc for T {}

pub trait CudaAlloc: private::alloc::Sealed {}
impl<T: private::alloc::Sealed> CudaAlloc for T {}

impl<T: CudaAlloc> private::alloc::Sealed for Option<T> {}

pub struct NullCudaAlloc;
impl private::alloc::Sealed for NullCudaAlloc {}
impl private::empty::Sealed for NullCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> private::empty::Sealed
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    pub fn new(front: A, tail: B) -> Self {
        Self(front, tail)
    }

    pub fn split(self) -> (A, B) {
        (self.0, self.1)
    }
}

pub struct CudaDropWrapper<C: private::drop::Sealed>(Option<C>);
impl<C: private::drop::Sealed> private::alloc::Sealed for CudaDropWrapper<C> {}
impl<C: private::drop::Sealed> From<C> for CudaDropWrapper<C> {
    fn from(val: C) -> Self {
        Self(Some(val))
    }
}
impl<C: private::drop::Sealed> Drop for CudaDropWrapper<C> {
    fn drop(&mut self) {
        if let Some(val) = self.0.take() {
            if let Err((_err, val)) = C::drop(val) {
                core::mem::forget(val);
            }
        }
    }
}
impl<C: private::drop::Sealed> Deref for CudaDropWrapper<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
impl<C: private::drop::Sealed> DerefMut for CudaDropWrapper<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

macro_rules! impl_sealed_drop_collection {
    ($type:ident) => {
        impl<C: DeviceCopy> private::drop::Sealed for $type<C> {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_collection!(DeviceBuffer);
impl_sealed_drop_collection!(DeviceBox);
impl_sealed_drop_collection!(LockedBuffer);

macro_rules! impl_sealed_drop_value {
    ($type:ident) => {
        impl private::drop::Sealed for $type {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_value!(Module);
impl_sealed_drop_value!(Stream);
impl_sealed_drop_value!(Context);

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct HostDeviceBox<T: DeviceCopy>(DevicePointer<T>);

impl<T: DeviceCopy> private::alloc::Sealed for HostDeviceBox<T> {}

impl<T: DeviceCopy> HostDeviceBox<T> {
    pub fn with_box<Q, F: FnOnce(&mut DeviceBox<T>) -> Q>(&mut self, inner: F) -> Q {
        // Safety: pointer comes from `DeviceBox::into_device`
        //         i.e. this function completes the roundtrip
        let mut device_box = unsafe { DeviceBox::from_device(self.0) };

        let result = inner(&mut device_box);

        self.0 = DeviceBox::into_device(device_box);

        result
    }
}

impl<T: DeviceCopy> From<DeviceBox<T>> for HostDeviceBox<T> {
    fn from(device_box: DeviceBox<T>) -> Self {
        Self(DeviceBox::into_device(device_box))
    }
}

impl<T: DeviceCopy> From<HostDeviceBox<T>> for DeviceBox<T> {
    fn from(host_device_box: HostDeviceBox<T>) -> Self {
        // Safety: pointer comes from `DeviceBox::into_device`
        //         i.e. this function completes the roundtrip
        unsafe { DeviceBox::from_device(host_device_box.0) }
    }
}

impl<T: DeviceCopy> Drop for HostDeviceBox<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from `DeviceBox::into_device`
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { DeviceBox::from_device(self.0) };

        core::mem::drop(CudaDropWrapper::from(device_box));
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRef<'a, T: DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_ref: &'a mut T,
}

impl<'a, T: DeviceCopy> HostAndDeviceMutRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(device_box: &'a mut HostDeviceBox<T>, host_ref: &'a mut T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub fn for_device(&mut self) -> DeviceMutRef<T> {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&mut self) -> &T {
        self.host_ref
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceConstRef<'a, T: DeviceCopy> {
    device_box: &'a HostDeviceBox<T>,
    host_ref: &'a T,
}

impl<'a, T: DeviceCopy> HostAndDeviceConstRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(device_box: &'a HostDeviceBox<T>, host_ref: &'a T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub fn for_device(&self) -> DeviceConstRef<T> {
        DeviceConstRef {
            pointer: self.device_box.0.as_raw(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_ref
    }
}
