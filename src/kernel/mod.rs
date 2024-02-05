#[cfg(feature = "host")]
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::NonNull,
};

#[cfg(feature = "host")]
use rustacuda::{
    error::{CudaError, CudaResult},
    function::Function,
    module::Module,
};

#[cfg(feature = "kernel")]
pub use rust_cuda_kernel::kernel;

#[doc(hidden)]
#[cfg(all(feature = "kernel", feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub use rust_cuda_kernel::{check_kernel, compile_kernel, specialise_kernel_entry_point};

#[cfg(feature = "host")]
mod ptx_jit;
#[cfg(feature = "host")]
use ptx_jit::{PtxJITCompiler, PtxJITResult};

#[cfg(feature = "host")]
use crate::host::Stream;
use crate::safety::PortableBitSemantics;

pub mod param;

mod sealed {
    #[doc(hidden)]
    pub trait Sealed {}

    #[cfg(feature = "host")]
    pub struct Token;
}

#[cfg(all(feature = "host", not(doc)))]
#[doc(hidden)]
pub trait WithNewAsync<
    'stream,
    P: ?Sized + CudaKernelParameter,
    O,
    E: From<rustacuda::error::CudaError>,
>
{
    fn with<'b>(self, param: P::AsyncHostType<'stream, 'b>) -> Result<O, E>
    where
        P: 'b;
}

#[cfg(all(feature = "host", not(doc)))]
impl<
        'stream,
        P: ?Sized + CudaKernelParameter,
        O,
        E: From<rustacuda::error::CudaError>,
        F: for<'b> FnOnce(P::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    > WithNewAsync<'stream, P, O, E> for F
{
    fn with<'b>(self, param: P::AsyncHostType<'stream, 'b>) -> Result<O, E>
    where
        P: 'b,
    {
        (self)(param)
    }
}

#[cfg(feature = "device")]
#[doc(hidden)]
pub trait WithFfiAsDevice<P: ?Sized + CudaKernelParameter, O> {
    fn with<'b>(self, param: P::DeviceType<'b>) -> O
    where
        P: 'b;
}

#[cfg(feature = "device")]
impl<P: ?Sized + CudaKernelParameter, O, F: for<'b> FnOnce(P::DeviceType<'b>) -> O>
    WithFfiAsDevice<P, O> for F
{
    fn with<'b>(self, param: P::DeviceType<'b>) -> O
    where
        P: 'b,
    {
        (self)(param)
    }
}

pub trait CudaKernelParameter: sealed::Sealed {
    #[cfg(feature = "host")]
    type SyncHostType;
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b>
    where
        Self: 'b;
    #[doc(hidden)]
    type FfiType<'stream, 'b>: PortableBitSemantics
    where
        Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b>
    where
        Self: 'b;

    #[cfg(feature = "host")]
    #[allow(clippy::missing_errors_doc)] // FIXME
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        param: &Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        param: &Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> std::alloc::Layout
    where
        Self: 'b;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b;

    #[doc(hidden)]
    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short;
}

#[cfg(feature = "host")]
pub struct Launcher<'stream, 'kernel, Kernel> {
    pub stream: Stream<'stream>,
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
        ) -> CudaResult<crate::utils::r#async::Async<
            'static, 'stream, (), crate::utils::r#async::NoCompletion,
        >>
        where
            Kernel: FnOnce(&mut Launcher<'stream, '_, Kernel>, $($T),*),
        {
            self.kernel.$launch_async::<$($T),*>(self.stream, &self.config, $($arg),*)
        }
    };
    (impl $func:ident () + ($($other:expr),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:expr),*) $inner:block) => {
        $T0::$func($arg0 $(, $other)*, |$arg0: <$T0 as CudaKernelParameter>::AsyncHostType<'stream, '_>| {
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
        let module: Box<Module> = Box::new(Module::load_from_string(ptx)?);

        let function = unsafe { &*std::ptr::from_ref(module.as_ref()) }.get_function(entry_point);

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
            stream: Stream<'stream>,
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
                    let r#async = kernel.$launch_async::<$($T),*>(stream, config, $($arg),*)?;

                    // important: always synchronise here, this function is sync!
                    r#async.synchronize()
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
            stream: Stream<'stream>,
            config: &LaunchConfig,
            $($arg: $T::SyncHostType,)*
            inner: impl FnOnce(
                &'kernel mut Self,
                Stream<'stream>,
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
            stream: Stream<'stream>,
            config: &LaunchConfig,
            $($arg: $T::AsyncHostType<'stream, '_>),*
        ) -> CudaResult<crate::utils::r#async::Async<
            'static, 'stream, (), crate::utils::r#async::NoCompletion,
        >>
        // launch_async does not need to capture its parameters until kernel completion:
        //  - moved parameters are moved and cannot be used again, deallocation will sync
        //  - immutably borrowed parameters can be shared across multiple kernel launches
        //  - mutably borrowed parameters are more tricky:
        //    - Rust's borrowing rules ensure that a single mutable reference cannot be
        //      passed into multiple parameters of the kernel (no mutable aliasing)
        //    - CUDA guarantees that kernels launched on the same stream are executed
        //      sequentially, so even immediate resubmissions for the same mutable data
        //      will not have temporally overlapping mutation on the same stream
        //    - however, we have to guarantee that mutable data cannot be used on several
        //      different streams at the same time
        //      - Async::move_to_stream always adds a synchronisation barrier between the
        //        old and the new stream to ensure that all uses on the old stream happen
        //        strictly before all uses on the new stream
        //      - async launches take AsyncProj<&mut HostAndDeviceMutRef<..>>, which either
        //        captures an Async, which must be moved to a different stream explicitly,
        //        or contains data that cannot async move to a different stream without
        //      - any use of a mutable borrow in an async kernel launch adds a sync barrier
        //        on the launch stream s.t. the borrow is only complete once the kernel has
        //        completed
        where
            Kernel: FnOnce(&mut Launcher<'stream, 'kernel, Kernel>, $($T),*),
        {
            let function = if config.ptx_jit {
                impl_typed_kernel_launch! { impl with_async_as_ptx_jit ref ($($arg: $T),*) + (sealed::Token) {
                    self.compile_with_ptx_jit_args(Some(&[$($arg),*]))
                } }?
            } else {
                self.compile_with_ptx_jit_args(None)?
            };

            #[allow(unused_mut)]
            let mut shared_memory_size = crate::utils::shared::SharedMemorySize::new();
            $(
                shared_memory_size.add($T::shared_layout_for_async(&$arg, sealed::Token));
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
                        &mut $T::async_to_ffi($arg, sealed::Token)?
                    ).cast::<core::ffi::c_void>()),*
                ],
            ) }?;

            crate::utils::r#async::Async::pending(
                (), stream, crate::utils::r#async::NoCompletion,
            )
        }
    };
    (impl $func:ident () + ($($other:expr),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:expr),*) $inner:block) => {
        $T0::$func($arg0 $(, $other)*, |$arg0: <$T0 as CudaKernelParameter>::AsyncHostType<'stream, '_>| {
            impl_typed_kernel_launch! { impl $func ($($arg: $T),*) + ($($other),*) $inner }
        })
    };
    (impl $func:ident ref () + ($($other:expr),*) $inner:block) => {
        $inner
    };
    (impl $func:ident ref ($arg0:ident : $T0:ident $(, $arg:ident : $T:ident)*) + ($($other:expr),*) $inner:block) => {
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
