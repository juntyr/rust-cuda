#[cfg(feature = "host")]
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::NonNull,
};

use const_type_layout::TypeGraphLayout;
#[cfg(feature = "host")]
use rustacuda::{
    error::{CudaError, CudaResult},
    function::Function,
    module::Module,
    stream::Stream,
};

#[cfg(feature = "derive")]
pub use rust_cuda_derive::kernel;

#[doc(hidden)]
#[cfg(all(feature = "derive", feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub use rust_cuda_derive::{check_kernel, link_kernel, specialise_kernel_entry_point};

#[cfg(feature = "host")]
mod ptx_jit;
#[cfg(feature = "host")]
use ptx_jit::{PtxJITCompiler, PtxJITResult};

pub mod param;

mod sealed {
    #[doc(hidden)]
    pub trait Sealed {}
}

pub trait CudaKernelParameter: sealed::Sealed {
    #[cfg(feature = "host")]
    type SyncHostType;
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b>;
    #[doc(hidden)]
    type FfiType<'stream, 'b>: rustacuda_core::DeviceCopy + TypeGraphLayout;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b>;

    #[cfg(feature = "host")]
    #[allow(clippy::missing_errors_doc)] // FIXME
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn shared_layout_for_async(param: &Self::AsyncHostType<'_, '_>) -> std::alloc::Layout;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b>;

    #[doc(hidden)]
    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O;
}

#[cfg(feature = "host")]
pub struct Launcher<'stream, 'kernel, Kernel> {
    pub stream: &'stream Stream,
    pub kernel: &'kernel mut TypedPtxKernel<Kernel>,
    pub config: LaunchConfig,
}

#[cfg(feature = "host")]
macro_rules! impl_launcher_launch {
    ($launch:ident($($arg:ident : $T:ident),*) => $with_async:ident => $launch_async:ident) => {
        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch<$($T: CudaKernelParameter),*>(
            &mut self,
            $($arg: $T::SyncHostType),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<'stream, '_, Kernel>, $($T),*),
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
            Kernel: FnOnce(&mut Launcher<'stream, '_, Kernel>, $($T),*),
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
            Kernel: FnOnce(&mut Launcher<'stream, '_, Kernel>, $($T),*),
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

#[cfg(feature = "host")]
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

#[cfg(feature = "host")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: rustacuda::function::GridSize,
    pub block: rustacuda::function::BlockSize,
    pub ptx_jit: bool,
}

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub struct RawPtxKernel {
    module: ManuallyDrop<Box<Module>>,
    function: ManuallyDrop<Function<'static>>,
}

#[cfg(feature = "host")]
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

#[cfg(feature = "host")]
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

#[cfg(feature = "host")]
pub type PtxKernelConfigure = dyn FnMut(&Function) -> CudaResult<()>;

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub struct TypedPtxKernel<Kernel> {
    compiler: PtxJITCompiler,
    ptx_kernel: Option<RawPtxKernel>,
    entry_point: Box<CStr>,
    configure: Option<Box<PtxKernelConfigure>>,
    marker: PhantomData<Kernel>,
}

#[cfg(feature = "host")]
macro_rules! impl_typed_kernel_launch {
    ($launch:ident($($arg:ident : $T:ident),*) => $with_async:ident => $launch_async:ident) => {
        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch<'kernel, 'stream, $($T: CudaKernelParameter),*>(
            &'kernel mut self,
            stream: &'stream Stream,
            config: &LaunchConfig,
            $($arg: $T::SyncHostType),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<'stream, 'kernel, Kernel>, $($T),*),
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
            'kernel,
            'stream,
            Ok,
            Err: From<CudaError>,
            $($T: CudaKernelParameter),*
        >(
            &'kernel mut self,
            stream: &'stream Stream,
            config: &LaunchConfig,
            $($arg: $T::SyncHostType,)*
            inner: impl FnOnce(
                &'kernel mut Self,
                &'stream Stream,
                &LaunchConfig,
                $($T::AsyncHostType<'stream, '_>),*
            ) -> Result<Ok, Err>,
        ) -> Result<Ok, Err>
        where
            Kernel: FnOnce(&mut Launcher<'stream, 'kernel, Kernel>, $($T),*),
        {
            impl_typed_kernel_launch! { impl with_new_async ($($arg: $T),*) + (stream) {
                inner(self, stream, config, $($arg),*)
            } }
        }

        #[allow(clippy::missing_errors_doc)]
        #[allow(clippy::needless_lifetimes)] // 'stream is unused for zero args
        #[allow(clippy::too_many_arguments)] // func is defined for <= 12 args
        pub fn $launch_async<'kernel, 'stream, $($T: CudaKernelParameter),*>(
            &'kernel mut self,
            stream: &'stream Stream,
            config: &LaunchConfig,
            $($arg: $T::AsyncHostType<'stream, '_>),*
        ) -> CudaResult<()>
        where
            Kernel: FnOnce(&mut Launcher<'stream, 'kernel, Kernel>, $($T),*),
        {
            let function = if config.ptx_jit {
                impl_typed_kernel_launch! { impl with_async_as_ptx_jit ref ($($arg: $T),*) + () {
                    self.compile_with_ptx_jit_args(Some(&[$($arg),*]))
                } }?
            } else {
                self.compile_with_ptx_jit_args(None)?
            };

            #[allow(unused_mut)]
            let mut shared_memory_size = crate::utils::shared::SharedMemorySize::new();
            $(
                shared_memory_size.add($T::shared_layout_for_async(&$arg));
            )*
            let Ok(shared_memory_size) = u32::try_from(shared_memory_size.total()) else {
                // FIXME: this should really be InvalidConfiguration = 9
                return Err(CudaError::LaunchOutOfResources)
            };

            unsafe { stream.launch(
                function,
                config.grid.clone(),
                config.block.clone(),
                shared_memory_size,
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

#[cfg(feature = "host")]
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

#[cfg(feature = "host")]
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

#[cfg(feature = "host")]
/// # Safety
///
/// The PTX string returned by [`CompiledKernelPtx::get_ptx`] must correspond
/// to the compiled kernel code for the `Kernel` function and contain a kernel
/// entry point whose name is returned by
/// [`CompiledKernelPtx::get_entry_point`].
///
/// This trait should not be implemented manually &ndash; use the
/// [`kernel`] macro instead.
pub unsafe trait CompiledKernelPtx<Kernel> {
    fn get_ptx() -> &'static CStr;
    fn get_entry_point() -> &'static CStr;
}
