use core::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
    event::Event,
    function::Function,
    memory::{DeviceBox, DeviceBuffer, LockedBox, LockedBuffer},
    module::Module,
    stream::Stream,
};
use rustacuda_core::{DeviceCopy, DevicePointer};

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{check_kernel, link_kernel, specialise_kernel_call};

use crate::{
    common::{
        DeviceAccessible, DeviceConstRef, DeviceMutRef, EmptyCudaAlloc, NoCudaAlloc, RustToCuda,
    },
    ptx_jit::{CudaKernel, PtxJITCompiler, PtxJITResult},
    safety::SafeDeviceCopy,
};

pub trait Launcher {
    type KernelTraitObject: ?Sized;
    type CompilationWatcher<'a>;

    fn get_launch_package(&mut self) -> LaunchPackage<Self>;

    /// # Errors
    ///
    /// Should only return a [`CudaError`] if some implementation-defined
    ///  critical kernel function configuration failed.
    #[allow(unused_variables)]
    fn on_compile(kernel: &Function, watcher: Self::CompilationWatcher<'_>) -> CudaResult<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: rustacuda::function::GridSize,
    pub block: rustacuda::function::BlockSize,
    pub shared_memory_size: u32,
    pub ptx_jit: bool,
}

pub struct LaunchPackage<'l, L: ?Sized + Launcher> {
    pub config: LaunchConfig,
    pub kernel: &'l mut TypedKernel<L::KernelTraitObject>,
    pub watcher: L::CompilationWatcher<'l>,
}

pub struct SimpleKernelLauncher<KernelTraitObject: ?Sized> {
    pub kernel: TypedKernel<KernelTraitObject>,
    pub config: LaunchConfig,
}

impl<KernelTraitObject: ?Sized> Launcher for SimpleKernelLauncher<KernelTraitObject> {
    type CompilationWatcher<'a> = ();
    type KernelTraitObject = KernelTraitObject;

    fn get_launch_package(&mut self) -> LaunchPackage<Self> {
        LaunchPackage {
            config: self.config.clone(),
            kernel: &mut self.kernel,
            watcher: (),
        }
    }
}

pub enum KernelJITResult<'k> {
    Cached(&'k Function<'k>),
    Recompiled(&'k Function<'k>),
}

pub struct TypedKernel<KernelTraitObject: ?Sized> {
    compiler: PtxJITCompiler,
    kernel: Option<CudaKernel>,
    entry_point: alloc::boxed::Box<std::ffi::CStr>,
    marker: PhantomData<KernelTraitObject>,
}

impl<KernelTraitObject: ?Sized> TypedKernel<KernelTraitObject> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] if `ptx` or `entry_point` contain nul bytes.
    pub fn new(ptx: &str, entry_point: &str) -> CudaResult<Self> {
        let ptx_cstring = std::ffi::CString::new(ptx).map_err(|_| CudaError::InvalidPtx)?;

        let compiler = crate::ptx_jit::PtxJITCompiler::new(&ptx_cstring);

        let entry_point_cstring =
            std::ffi::CString::new(entry_point).map_err(|_| CudaError::InvalidValue)?;
        let entry_point = entry_point_cstring.into_boxed_c_str();

        Ok(Self {
            compiler,
            kernel: None,
            entry_point,
            marker: PhantomData,
        })
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] if `ptx` (from [`Self::new`]) is not a valid
    ///  PTX source, or it does not contain an entry point named `entry_point`
    ///  (from [`Self::new`]).
    pub fn compile_with_ptx_jit_args(
        &mut self,
        arguments: Option<&[Option<*const [u8]>]>,
    ) -> CudaResult<KernelJITResult> {
        let ptx_jit = self.compiler.with_arguments(arguments);

        let kernel_jit = match (&mut self.kernel, ptx_jit) {
            (Some(kernel), PtxJITResult::Cached(_)) => {
                KernelJITResult::Cached(kernel.get_function())
            },
            (kernel, PtxJITResult::Cached(ptx_cstr) | PtxJITResult::Recomputed(ptx_cstr)) => {
                let recomputed_kernel = CudaKernel::new(ptx_cstr, &self.entry_point)?;

                // Replace the existing compiled kernel, drop the old one
                let kernel = kernel.insert(recomputed_kernel);

                KernelJITResult::Recompiled(kernel.get_function())
            },
        };

        Ok(kernel_jit)
    }
}

pub trait LendToCuda: RustToCuda {
    /// Lends an immutable copy of `&self` to CUDA:
    /// - code in the CUDA kernel can only access `&self` through the
    ///   [`DeviceConstRef`] inside the closure
    /// - after the closure, `&self` will not have changed
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn lend_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &self,
        inner: F,
    ) -> Result<O, E>;

    /// Lends a mutable copy of `&mut self` to CUDA:
    /// - code in the CUDA kernel can only access `&mut self` through the
    ///   [`DeviceMutRef`] inside the closure
    /// - after the closure, `&mut self` might have changed in the following
    ///   ways:
    ///   - to avoid aliasing, each CUDA thread gets its own shallow copy of
    ///     `&mut self`, i.e. any shallow changes will NOT be reflected after
    ///     the closure
    ///   - each CUDA thread can access the same heap allocated storage, i.e.
    ///     any deep changes will be reflected after the closure
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn lend_to_cuda_mut<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &mut self,
        inner: F,
    ) -> Result<O, E>;

    /// Moves `self` to CUDA iff `self` is [`SafeDeviceCopy`]
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn move_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy,
        <Self as RustToCuda>::CudaAllocation: EmptyCudaAlloc;
}

impl<T: RustToCuda> LendToCuda for T {
    fn lend_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &self,
        inner: F,
    ) -> Result<O, E> {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceConstRef::with_new(&cuda_repr, inner);

        core::mem::drop(cuda_repr);
        core::mem::drop(alloc);

        result
    }

    fn lend_to_cuda_mut<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &mut self,
        inner: F,
    ) -> Result<O, E> {
        let (mut cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceMutRef::with_new(&mut cuda_repr, inner);

        core::mem::drop(cuda_repr);

        let _: NoCudaAlloc = unsafe { self.restore(alloc) }?;

        result
    }

    fn move_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy,
        <Self as RustToCuda>::CudaAllocation: EmptyCudaAlloc,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceOwned::with_new(cuda_repr, inner);

        core::mem::drop(alloc);

        result
    }
}

mod private {
    pub mod drop {
        pub trait Sealed: Sized {
            fn drop(val: Self) -> Result<(), (rustacuda::error::CudaError, Self)>;
        }
    }
}

#[repr(transparent)]
pub struct CudaDropWrapper<C: private::drop::Sealed>(ManuallyDrop<C>);
impl<C: private::drop::Sealed> crate::common::crate_private::alloc::Sealed for CudaDropWrapper<C> {}
impl<C: private::drop::Sealed> From<C> for CudaDropWrapper<C> {
    fn from(val: C) -> Self {
        Self(ManuallyDrop::new(val))
    }
}
impl<C: private::drop::Sealed> Drop for CudaDropWrapper<C> {
    fn drop(&mut self) {
        // Safety: drop is only ever called once
        let val = unsafe { ManuallyDrop::take(&mut self.0) };

        if let Err((_err, val)) = C::drop(val) {
            core::mem::forget(val);
        }
    }
}
impl<C: private::drop::Sealed> Deref for CudaDropWrapper<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<C: private::drop::Sealed> DerefMut for CudaDropWrapper<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
impl_sealed_drop_collection!(LockedBox);

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
impl_sealed_drop_value!(Event);

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct HostLockedBox<T: DeviceCopy>(*mut T);

impl<T: DeviceCopy> HostLockedBox<T> {
    /// # Errors
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        // Safety: uninitialised memory is immediately written to without reading it
        let locked_ptr = unsafe {
            let locked_ptr: *mut T = LockedBox::into_raw(LockedBox::uninitialized()?);
            locked_ptr.write(value);
            locked_ptr
        };

        Ok(Self(locked_ptr))
    }
}

impl<T: DeviceCopy> Deref for HostLockedBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T: DeviceCopy> DerefMut for HostLockedBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl<T: DeviceCopy> From<LockedBox<T>> for HostLockedBox<T> {
    fn from(locked_box: LockedBox<T>) -> Self {
        Self(LockedBox::into_raw(locked_box))
    }
}

impl<T: DeviceCopy> From<HostLockedBox<T>> for LockedBox<T> {
    fn from(host_locked_box: HostLockedBox<T>) -> Self {
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        unsafe { LockedBox::from_raw(host_locked_box.0) }
    }
}

impl<T: DeviceCopy> Drop for HostLockedBox<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let locked_box = unsafe { LockedBox::from_raw(self.0) };

        core::mem::drop(CudaDropWrapper::from(locked_box));
    }
}

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct HostDeviceBox<T: DeviceCopy>(DevicePointer<T>);

impl<T: DeviceCopy> crate::common::crate_private::alloc::Sealed for HostDeviceBox<T> {}

impl<T: DeviceCopy> HostDeviceBox<T> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `value` into `self` failed.
    pub fn copy_from(&mut self, value: &T) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let mut device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };

        rustacuda::memory::CopyDestination::copy_from(&mut *device_box, value)
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `self` into `value` failed.
    pub fn copy_to(&self, value: &mut T) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };

        rustacuda::memory::CopyDestination::copy_to(&*device_box, value)
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `value` into `self` failed.
    ///
    /// # Safety
    ///
    /// To use the data inside the device box, either
    /// - the passed-in [`Stream`] must be synchronised
    /// - the kernel must be launched on the passed-in [`Stream`]
    pub unsafe fn async_copy_from(
        &mut self,
        value: &HostLockedBox<T>,
        stream: &Stream,
    ) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let mut device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let locked_box = unsafe { ManuallyDrop::new(LockedBox::from_raw(value.0)) };

        unsafe {
            rustacuda::memory::AsyncCopyDestination::async_copy_from(
                &mut *device_box,
                &*locked_box,
                stream,
            )
        }
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `self` into `value` failed.
    ///
    /// # Safety
    ///
    /// To use the data inside `value`, the passed-in [`Stream`] must be
    /// synchronised.
    pub unsafe fn async_copy_to(
        &self,
        value: &mut HostLockedBox<T>,
        stream: &Stream,
    ) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let mut locked_box = unsafe { ManuallyDrop::new(LockedBox::from_raw(value.0)) };

        unsafe {
            rustacuda::memory::AsyncCopyDestination::async_copy_to(
                &*device_box,
                &mut *locked_box,
                stream,
            )
        }
    }
}

impl<T: DeviceCopy> From<DeviceBox<T>> for HostDeviceBox<T> {
    fn from(device_box: DeviceBox<T>) -> Self {
        Self(DeviceBox::into_device(device_box))
    }
}

impl<T: DeviceCopy> From<HostDeviceBox<T>> for DeviceBox<T> {
    fn from(host_device_box: HostDeviceBox<T>) -> Self {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        unsafe { DeviceBox::from_device(host_device_box.0) }
    }
}

impl<T: DeviceCopy> Drop for HostDeviceBox<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from [`DeviceBox::into_device`]
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

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(HostAndDeviceMutRef<'b, T>) -> Result<O, E>,
    >(
        host_ref: &mut T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box: HostDeviceBox<_> = DeviceBox::new(host_ref)?.into();

        // Safety: `device_box` contains exactly the device copy of `host_ref`
        let result = inner(HostAndDeviceMutRef {
            device_box: &mut device_box,
            host_ref,
        });

        // Copy back any changes made
        device_box.copy_to(host_ref)?;

        core::mem::drop(device_box);

        result
    }

    #[must_use]
    pub fn for_device<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host<'b: 'a>(&'b self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRef<'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRef {
            device_box: self.device_box,
            host_ref: self.host_ref,
        }
    }

    #[must_use]
    pub fn as_mut<'b>(&'b mut self) -> HostAndDeviceMutRef<'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRef {
            device_box: self.device_box,
            host_ref: self.host_ref,
        }
    }

    #[must_use]
    pub fn as_async<'stream, 'b>(&'b mut self) -> HostAndDeviceMutRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceConstRef<'a, T: DeviceCopy> {
    device_box: &'a HostDeviceBox<T>,
    host_ref: &'a T,
}

impl<'a, T: DeviceCopy> Clone for HostAndDeviceConstRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: DeviceCopy> Copy for HostAndDeviceConstRef<'a, T> {}

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

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(HostAndDeviceConstRef<'b, T>) -> Result<O, E>,
    >(
        host_ref: &T,
        inner: F,
    ) -> Result<O, E> {
        let device_box: HostDeviceBox<_> = DeviceBox::new(host_ref)?.into();

        // Safety: `device_box` contains exactly the device copy of `host_ref`
        let result = inner(HostAndDeviceConstRef {
            device_box: &device_box,
            host_ref,
        });

        core::mem::drop(device_box);

        result
    }

    #[must_use]
    pub fn for_device<'b>(&'b self) -> DeviceConstRef<'a, T>
    where
        'a: 'b,
    {
        DeviceConstRef {
            pointer: self.device_box.0.as_raw(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRef<'b, T>
    where
        'a: 'b,
    {
        *self
    }

    #[must_use]
    pub fn as_async<'stream, 'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceOwned<'a, T: SafeDeviceCopy + DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_val: &'a mut T,
}

impl<'a, T: SafeDeviceCopy + DeviceCopy> HostAndDeviceOwned<'a, T> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(HostAndDeviceOwned<'b, T>) -> Result<O, E>,
    >(
        mut value: T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box: HostDeviceBox<_> = DeviceBox::new(&value)?.into();

        // Safety: `device_box` contains exactly the device copy of `value`
        let result = inner(HostAndDeviceOwned {
            device_box: &mut device_box,
            host_val: &mut value,
        });

        core::mem::drop(device_box);
        core::mem::drop(value);

        result
    }

    #[must_use]
    pub fn for_device(self) -> DeviceMutRef<'a, T> {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&'a mut self) -> &'a T {
        self.host_val
    }

    #[must_use]
    pub fn as_async<'stream, 'b>(&'b mut self) -> HostAndDeviceOwnedAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceOwnedAsync {
            device_box: self.device_box,
            host_val: self.host_val,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRefAsync<'stream, 'a, T: DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_ref: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: DeviceCopy> HostAndDeviceMutRefAsync<'stream, 'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(
        device_box: &'a mut HostDeviceBox<T>,
        host_ref: &'a mut T,
        stream: &'stream Stream,
    ) -> Self {
        let _ = stream;

        Self {
            device_box,
            host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }

    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceMutRef`] must only be used on the constructed-with
    /// [`Stream`]
    pub unsafe fn for_device_async<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host<'b: 'a>(&'b self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: self.stream,
        }
    }

    #[must_use]
    pub fn as_mut<'b>(&'b mut self) -> HostAndDeviceMutRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: self.stream,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceConstRefAsync<'stream, 'a, T: DeviceCopy> {
    device_box: &'a HostDeviceBox<T>,
    host_ref: &'a T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: DeviceCopy> Clone for HostAndDeviceConstRefAsync<'stream, 'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'stream, 'a, T: DeviceCopy> Copy for HostAndDeviceConstRefAsync<'stream, 'a, T> {}

impl<'stream, 'a, T: DeviceCopy> HostAndDeviceConstRefAsync<'stream, 'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(
        device_box: &'a HostDeviceBox<T>,
        host_ref: &'a T,
        stream: &'stream Stream,
    ) -> Self {
        let _ = stream;

        Self {
            device_box,
            host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }

    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceConstRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async<'b>(&'b self) -> DeviceConstRef<'a, T>
    where
        'a: 'b,
    {
        DeviceConstRef {
            pointer: self.device_box.0.as_raw(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        *self
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceOwnedAsync<'stream, 'a, T: SafeDeviceCopy + DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_val: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: SafeDeviceCopy + DeviceCopy> HostAndDeviceOwnedAsync<'stream, 'a, T> {
    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceConstRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async(self) -> DeviceMutRef<'a, T> {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host(&'a mut self) -> &'a T {
        self.host_val
    }
}
