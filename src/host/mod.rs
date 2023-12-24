use core::ptr::NonNull;
use std::{
    ffi::{CStr, CString},
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
pub use rust_cuda_derive::{check_kernel, link_kernel, specialise_kernel_entry_point};

use crate::{
    common::{
        CudaKernelParameter, DeviceAccessible, DeviceConstRef, DeviceMutRef, DeviceOwnedRef,
        EmptyCudaAlloc, NoCudaAlloc, RustToCuda,
    },
    safety::{NoSafeAliasing, SafeDeviceCopy},
};

mod ptx_jit;
use ptx_jit::{PtxJITCompiler, PtxJITResult};

pub struct Launcher<'stream, 'kernel, Kernel> {
    pub stream: &'stream Stream,
    pub kernel: &'kernel mut TypedPtxKernel<Kernel>,
    pub config: LaunchConfig,
}

macro_rules! impl_launcher_launch {
    ($launch:ident($($arg:ident : $T:ident),*) => $with_async:ident => $launch_async:ident) => {
        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch<$($T: CudaKernelParameter),*>(
            &mut self,
            $($arg: $T::SyncHostType),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            self.kernel.$launch::<$($T),*>(self.stream, &self.config, $($arg),*)
        }

        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $with_async<
            'a,
            Ok,
            Err: From<CudaError>,
            $($T: CudaKernelParameter),*
        >(
            &'a mut self,
            $($arg: $T::SyncHostType,)*
            inner: impl FnOnce(
                &'a mut Self,
                $($T::AsyncHostType<'stream, '_>),*
            ) -> Result<Ok, Err>,
        ) -> Result<Ok, Err>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            #[allow(unused_variables)]
            let stream = self.stream;

            impl_launcher_launch! { impl with_new_async ($($arg: $T),*) + (stream) {
                inner(self, $($arg),*)
            } }
        }

        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch_async<$($T: CudaKernelParameter),*>(
            &mut self,
            $($arg: $T::AsyncHostType<'stream, '_>),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            self.kernel.$launch_async::<$($T),*>(self.stream, &self.config, $($arg),*)
        }
    };
    (impl $func:ident () + ($($other:ident),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:ident),*) $inner:block) => {
        $T0::$func($arg0 $(, $other)*, |$arg0| {
            impl_launcher_launch! { impl $func ($($arg: $T),*) + ($($other),*) $inner }
        })
    };
}

impl<'stream, 'kernel, Kernel> Launcher<'stream, 'kernel, Kernel> {
    impl_launcher_launch! { launch0() => with0_async => launch0_async }

    impl_launcher_launch! { launch1(
        arg1: A
    ) => with1_async => launch1_async }

    impl_launcher_launch! { launch2(
        arg1: A, arg2: B
    ) => with2_async => launch2_async }

    impl_launcher_launch! { launch3(
        arg1: A, arg2: B, arg3: C
    ) => with3_async => launch3_async }

    impl_launcher_launch! { launch4(
        arg1: A, arg2: B, arg3: C, arg4: D
    ) => with4_async => launch4_async }

    impl_launcher_launch! { launch5(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E
    ) => with5_async => launch5_async }

    impl_launcher_launch! { launch6(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F
    ) => with6_async => launch6_async }

    impl_launcher_launch! { launch7(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G
    ) => with7_async => launch7_async }

    impl_launcher_launch! { launch8(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H
    ) => with8_async => launch8_async }

    impl_launcher_launch! { launch9(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I
    ) => with9_async => launch9_async }

    impl_launcher_launch! { launch10(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J
    ) => with10_async => launch10_async }

    impl_launcher_launch! { launch11(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J,
        arg11: K
    ) => with11_async => launch11_async }

    impl_launcher_launch! { launch12(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J,
        arg11: K, arg12: L
    ) => with12_async => launch12_async }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: rustacuda::function::GridSize,
    pub block: rustacuda::function::BlockSize,
    pub shared_memory_size: u32,
    pub ptx_jit: bool,
}

#[doc(cfg(feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub struct RawPtxKernel {
    module: ManuallyDrop<Box<Module>>,
    function: ManuallyDrop<Function<'static>>,
}

impl RawPtxKernel {
    /// # Errors
    ///
    /// Returns a [`CudaError`] if `ptx` is not a valid PTX source, or it does
    ///  not contain an entry point named `entry_point`.
    pub fn new(ptx: &CStr, entry_point: &CStr) -> CudaResult<Self> {
        let module = Box::new(Module::load_from_string(ptx)?);

        let function = unsafe { &*(module.as_ref() as *const Module) }.get_function(entry_point);

        let function = match function {
            Ok(function) => function,
            Err(err) => {
                if let Err((_err, module)) = Module::drop(*module) {
                    std::mem::forget(module);
                }

                return Err(err);
            },
        };

        Ok(Self {
            function: ManuallyDrop::new(function),
            module: ManuallyDrop::new(module),
        })
    }

    #[must_use]
    pub fn get_function(&self) -> &Function {
        &self.function
    }
}

impl Drop for RawPtxKernel {
    fn drop(&mut self) {
        {
            // Ensure that self.function is dropped before self.module as
            //  it borrows data from the module and must not outlive it
            let _function = unsafe { ManuallyDrop::take(&mut self.function) };
        }

        if let Err((_err, module)) = Module::drop(*unsafe { ManuallyDrop::take(&mut self.module) })
        {
            std::mem::forget(module);
        }
    }
}

pub type PtxKernelConfigure = dyn FnMut(&Function) -> CudaResult<()>;

pub struct TypedPtxKernel<Kernel> {
    compiler: PtxJITCompiler,
    ptx_kernel: Option<RawPtxKernel>,
    entry_point: Box<CStr>,
    configure: Option<Box<PtxKernelConfigure>>,
    marker: PhantomData<Kernel>,
}

macro_rules! impl_typed_kernel_launch {
    ($launch:ident($($arg:ident : $T:ident),*) => $with_async:ident => $launch_async:ident) => {
        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch<$($T: CudaKernelParameter),*>(
            &mut self,
            stream: &Stream,
            config: &LaunchConfig,
            $($arg: $T::SyncHostType),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            self.$with_async::<(), CudaError, $($T),*>(
                stream,
                config,
                $($arg,)*
                |kernel, stream, config, $($arg),*| {
                    let result = kernel.$launch_async::<$($T),*>(stream, config, $($arg),*);

                    // important: always synchronise here, this function is sync!
                    match (stream.synchronize(), result) {
                        (Ok(()), result) => result,
                        (Err(_), Err(err)) | (Err(err), Ok(())) => Err(err),
                    }
                },
            )
        }

        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $with_async<
            'a,
            'stream,
            Ok,
            Err: From<CudaError>,
            $($T: CudaKernelParameter),*
        >(
            &'a mut self,
            stream: &'stream Stream,
            config: &LaunchConfig,
            $($arg: $T::SyncHostType,)*
            inner: impl FnOnce(
                &'a mut Self,
                &'stream Stream,
                &LaunchConfig,
                $($T::AsyncHostType<'stream, '_>),*
            ) -> Result<Ok, Err>,
        ) -> Result<Ok, Err>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            impl_typed_kernel_launch! { impl with_new_async ($($arg: $T),*) + (stream) {
                inner(self, stream, config, $($arg),*)
            } }
        }

        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::needless_lifetimes)] // 'stream is unused for zero args
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch_async<'stream, $($T: CudaKernelParameter),*>(
            &mut self,
            stream: &'stream Stream,
            config: &LaunchConfig,
            $($arg: $T::AsyncHostType<'stream, '_>),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<Kernel>, $($T),*),
        {
            let function = if config.ptx_jit {
                impl_typed_kernel_launch! { impl with_async_as_ptx_jit ref ($($arg: $T),*) + () {
                    self.compile_with_ptx_jit_args(Some(&[$($arg),*]))
                } }?
            } else {
                self.compile_with_ptx_jit_args(None)?
            };

            unsafe { stream.launch(
                function,
                config.grid.clone(),
                config.block.clone(),
                config.shared_memory_size,
                &[
                    $(core::ptr::from_mut(
                        &mut $T::async_to_ffi($arg)
                    ).cast::<core::ffi::c_void>()),*
                ],
            ) }
        }
    };
    (impl $func:ident () + ($($other:ident),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:ident),*) $inner:block) => {
        $T0::$func($arg0 $(, $other)*, |$arg0| {
            impl_typed_kernel_launch! { impl $func ($($arg: $T),*) + ($($other),*) $inner }
        })
    };
    (impl $func:ident ref () + ($($other:ident),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ref ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:ident),*) $inner:block) => {
        $T0::$func(&$arg0 $(, $other)*, |$arg0| {
            impl_typed_kernel_launch! { impl $func ref ($($arg: $T),*) + ($($other),*) $inner }
        })
    };
}

impl<Kernel> TypedPtxKernel<Kernel> {
    #[must_use]
    pub fn new<T: CompiledKernelPtx<Kernel>>(configure: Option<Box<PtxKernelConfigure>>) -> Self {
        let compiler = PtxJITCompiler::new(T::get_ptx());
        let entry_point = CString::from(T::get_entry_point()).into_boxed_c_str();

        Self {
            compiler,
            ptx_kernel: None,
            entry_point,
            configure,
            marker: PhantomData::<Kernel>,
        }
    }
}

impl<Kernel> TypedPtxKernel<Kernel> {
    impl_typed_kernel_launch! { launch0() => with0_async => launch0_async }

    impl_typed_kernel_launch! { launch1(
        arg1: A
    ) => with1_async => launch1_async }

    impl_typed_kernel_launch! { launch2(
        arg1: A, arg2: B
    ) => with2_async => launch2_async }

    impl_typed_kernel_launch! { launch3(
        arg1: A, arg2: B, arg3: C
    ) => with3_async => launch3_async }

    impl_typed_kernel_launch! { launch4(
        arg1: A, arg2: B, arg3: C, arg4: D
    ) => with4_async => launch4_async }

    impl_typed_kernel_launch! { launch5(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E
    ) => with5_async => launch5_async }

    impl_typed_kernel_launch! { launch6(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F
    ) => with6_async => launch6_async }

    impl_typed_kernel_launch! { launch7(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G
    ) => with7_async => launch7_async }

    impl_typed_kernel_launch! { launch8(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H
    ) => with8_async => launch8_async }

    impl_typed_kernel_launch! { launch9(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I
    ) => with9_async => launch9_async }

    impl_typed_kernel_launch! { launch10(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J
    ) => with10_async => launch10_async }

    impl_typed_kernel_launch! { launch11(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J,
        arg11: K
    ) => with11_async => launch11_async }

    impl_typed_kernel_launch! { launch12(
        arg1: A, arg2: B, arg3: C, arg4: D, arg5: E, arg6: F, arg7: G, arg8: H, arg9: I, arg10: J,
        arg11: K, arg12: L
    ) => with12_async => launch12_async }

    /// # Errors
    ///
    /// Returns a [`CudaError`] if the [`CompiledKernelPtx`] provided to
    /// [`Self::new`] is not a valid PTX source or does not contain the
    /// entry point it declares.
    fn compile_with_ptx_jit_args(
        &mut self,
        arguments: Option<&[Option<&NonNull<[u8]>>]>,
    ) -> CudaResult<&Function> {
        let ptx_jit = self.compiler.with_arguments(arguments);

        let kernel_jit = match (&mut self.ptx_kernel, ptx_jit) {
            (Some(ptx_kernel), PtxJITResult::Cached(_)) => ptx_kernel.get_function(),
            (ptx_kernel, PtxJITResult::Cached(ptx_cstr) | PtxJITResult::Recomputed(ptx_cstr)) => {
                let recomputed_ptx_kernel = RawPtxKernel::new(ptx_cstr, &self.entry_point)?;

                // Replace the existing compiled kernel, drop the old one
                let ptx_kernel = ptx_kernel.insert(recomputed_ptx_kernel);

                let function = ptx_kernel.get_function();

                if let Some(configure) = self.configure.as_mut() {
                    configure(function)?;
                }

                function
            },
        };

        Ok(kernel_jit)
    }
}

pub trait LendToCuda: RustToCuda + NoSafeAliasing {
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
        Self: RustToCuda<CudaRepresentation: SafeDeviceCopy, CudaAllocation: EmptyCudaAlloc>;
}

impl<T: RustToCuda + NoSafeAliasing> LendToCuda for T {
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
        Self: RustToCuda<CudaRepresentation: SafeDeviceCopy, CudaAllocation: EmptyCudaAlloc>,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceOwned::with_new(cuda_repr, inner);

        core::mem::drop(alloc);

        result
    }
}

pub trait CudaDroppable: Sized {
    #[allow(clippy::missing_errors_doc)]
    fn drop(val: Self) -> Result<(), (rustacuda::error::CudaError, Self)>;
}

#[repr(transparent)]
pub struct CudaDropWrapper<C: CudaDroppable>(ManuallyDrop<C>);
impl<C: CudaDroppable> crate::common::CudaAlloc for CudaDropWrapper<C> {}
impl<C: CudaDroppable> crate::common::crate_private::alloc::Sealed for CudaDropWrapper<C> {}
impl<C: CudaDroppable> From<C> for CudaDropWrapper<C> {
    fn from(val: C) -> Self {
        Self(ManuallyDrop::new(val))
    }
}
impl<C: CudaDroppable> Drop for CudaDropWrapper<C> {
    fn drop(&mut self) {
        // Safety: drop is only ever called once
        let val = unsafe { ManuallyDrop::take(&mut self.0) };

        if let Err((_err, val)) = C::drop(val) {
            core::mem::forget(val);
        }
    }
}
impl<C: CudaDroppable> Deref for CudaDropWrapper<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<C: CudaDroppable> DerefMut for CudaDropWrapper<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! impl_sealed_drop_collection {
    ($type:ident) => {
        impl<C: DeviceCopy> CudaDroppable for $type<C> {
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
        impl CudaDroppable for $type {
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
        unsafe { Self::from_raw(host_locked_box.0) }
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

impl<T: DeviceCopy> crate::common::CudaAlloc for HostDeviceBox<T> {}
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
        unsafe { Self::from_device(host_device_box.0) }
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
    pub const unsafe fn new(device_box: &'a HostDeviceBox<T>, host_ref: &'a T) -> Self {
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
    pub const fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub const fn as_ref<'b>(&'b self) -> HostAndDeviceConstRef<'b, T>
    where
        'a: 'b,
    {
        *self
    }

    #[must_use]
    pub const fn as_async<'stream, 'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
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
    pub fn with_new<O, E: From<CudaError>, F: FnOnce(HostAndDeviceOwned<T>) -> Result<O, E>>(
        mut value: T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box: HostDeviceBox<_> = DeviceBox::new(&value)?.into();

        // Safety: `device_box` contains exactly the device copy of `value`
        inner(HostAndDeviceOwned {
            device_box: &mut device_box,
            host_val: &mut value,
        })
    }

    #[must_use]
    pub fn for_device(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: self.device_box.0.as_raw_mut(),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }

    #[must_use]
    pub fn into_async<'stream>(self) -> HostAndDeviceOwnedAsync<'stream, 'a, T> {
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
    #[must_use]
    pub const unsafe fn new(
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
    pub const fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub const fn as_ref<'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
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
    /// The returned [`DeviceOwnedRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: self.device_box.0.as_raw_mut(),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }
}

/// # Safety
///
/// The PTX string returned by [`CompiledKernelPtx::get_ptx`] must correspond
/// to the compiled kernel code for the `Kernel` function and contain a kernel
/// entry point whose name is returned by
/// [`CompiledKernelPtx::get_entry_point`].
///
/// This trait should not be implemented manually &ndash; use the
/// [`kernel`](crate::common::kernel) macro instead.
pub unsafe trait CompiledKernelPtx<Kernel> {
    fn get_ptx() -> &'static CStr;
    fn get_entry_point() -> &'static CStr;
}
