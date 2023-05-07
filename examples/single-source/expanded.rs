#![feature(prelude_import)]
#![deny(clippy::pedantic)]
#![feature(cfg_version)]
#![feature(const_type_name)]
#![feature(const_refs_to_cell)]
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
extern crate alloc;
#[cfg(not(target_os = "cuda"))]
fn main() {}
#[repr(C)]
#[layout(crate = "rc::const_type_layout")]
pub struct Dummy(i32);
unsafe impl const rc::const_type_layout::TypeLayout for Dummy {
    const TYPE_LAYOUT: rc::const_type_layout::TypeLayoutInfo<'static> = {
        rc::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: rc::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: &[
                    rc::const_type_layout::Field {
                        name: "0",
                        offset: {
                            {
                                #[allow(clippy::unneeded_field_pattern)]
                                let Dummy { 0: _, .. }: Dummy;
                                if let ::const_type_layout::MaybeUninhabited::Inhabited(
                                    uninit,
                                )
                                    = unsafe {
                                        <Dummy as ::const_type_layout::TypeLayout>::uninit()
                                    } {
                                    let base_ptr: *const Dummy = (&raw const uninit).cast();
                                    #[allow(unused_unsafe)]
                                    let field_ptr = unsafe { &raw const (*base_ptr).0 };
                                    #[allow(clippy::cast_sign_loss)]
                                    let offset = unsafe {
                                        field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                                    };
                                    #[allow(clippy::forget_non_drop, clippy::forget_copy)]
                                    core::mem::forget(uninit);
                                    ::const_type_layout::MaybeUninhabited::Inhabited(offset)
                                } else {
                                    ::const_type_layout::MaybeUninhabited::Uninhabited
                                }
                            }
                        },
                        ty: ::core::any::type_name::<i32>(),
                    },
                ],
            },
        }
    };
    unsafe fn uninit() -> rc::const_type_layout::MaybeUninhabited<
        ::core::mem::MaybeUninit<Self>,
    > {
        if let (rc::const_type_layout::MaybeUninhabited::Inhabited(f_0))
            = (<i32 as rc::const_type_layout::TypeLayout>::uninit()) {
            rc::const_type_layout::MaybeUninhabited::Inhabited(
                ::core::mem::MaybeUninit::new(Dummy(f_0.assume_init())),
            )
        } else {
            rc::const_type_layout::MaybeUninhabited::Uninhabited
        }
    }
}
unsafe impl const rc::const_type_layout::TypeGraph for Dummy {
    fn populate_graph(graph: &mut rc::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as rc::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <i32 as rc::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
#[cuda(crate = "rc")]
#[allow(dead_code)]
pub struct Wrapper<T> {
    #[cuda(embed)]
    inner: T,
}
#[allow(dead_code)]
#[doc(hidden)]
#[allow(dead_code)]
#[repr(C)]
#[layout(free = "T")]
#[layout(crate = "rc :: const_type_layout")]
pub struct WrapperCudaRepresentation<T>
where
    T: rc::common::RustToCuda,
{
    inner: rc::common::DeviceAccessible<
        <T as rc::common::RustToCuda>::CudaRepresentation,
    >,
}
unsafe impl<T> const rc::const_type_layout::TypeLayout for WrapperCudaRepresentation<T>
where
    T: rc::common::RustToCuda,
{
    const TYPE_LAYOUT: rc::const_type_layout::TypeLayoutInfo<'static> = {
        rc::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: rc::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: &[
                    rc::const_type_layout::Field {
                        name: "inner",
                        offset: {
                            {
                                #[allow(clippy::unneeded_field_pattern)]
                                let WrapperCudaRepresentation {
                                    inner: _,
                                    ..
                                }: WrapperCudaRepresentation<T>;
                                if let ::const_type_layout::MaybeUninhabited::Inhabited(
                                    uninit,
                                )
                                    = unsafe {
                                        <WrapperCudaRepresentation<
                                            T,
                                        > as ::const_type_layout::TypeLayout>::uninit()
                                    } {
                                    let base_ptr: *const WrapperCudaRepresentation<T> = (&raw const uninit)
                                        .cast();
                                    #[allow(unused_unsafe)]
                                    let field_ptr = unsafe { &raw const (*base_ptr).inner };
                                    #[allow(clippy::cast_sign_loss)]
                                    let offset = unsafe {
                                        field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                                    };
                                    #[allow(clippy::forget_non_drop, clippy::forget_copy)]
                                    core::mem::forget(uninit);
                                    ::const_type_layout::MaybeUninhabited::Inhabited(offset)
                                } else {
                                    ::const_type_layout::MaybeUninhabited::Uninhabited
                                }
                            }
                        },
                        ty: ::core::any::type_name::<
                            rc::common::DeviceAccessible<
                                <T as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >(),
                    },
                ],
            },
        }
    };
    unsafe fn uninit() -> rc::const_type_layout::MaybeUninhabited<
        ::core::mem::MaybeUninit<Self>,
    > {
        if let (rc::const_type_layout::MaybeUninhabited::Inhabited(inner))
            = (<rc::common::DeviceAccessible<
                <T as rc::common::RustToCuda>::CudaRepresentation,
            > as rc::const_type_layout::TypeLayout>::uninit()) {
            rc::const_type_layout::MaybeUninhabited::Inhabited(
                ::core::mem::MaybeUninit::new(WrapperCudaRepresentation {
                    inner: inner.assume_init(),
                }),
            )
        } else {
            rc::const_type_layout::MaybeUninhabited::Uninhabited
        }
    }
}
unsafe impl<T> const rc::const_type_layout::TypeGraph for WrapperCudaRepresentation<T>
where
    T: rc::common::RustToCuda,
{
    fn populate_graph(graph: &mut rc::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as rc::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <rc::common::DeviceAccessible<
                <T as rc::common::RustToCuda>::CudaRepresentation,
            > as rc::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
unsafe impl<T> rc::rustacuda_core::DeviceCopy for WrapperCudaRepresentation<T>
where
    T: rc::common::RustToCuda,
{}
unsafe impl<T> rc::common::RustToCuda for Wrapper<T>
where
    T: rc::common::RustToCuda,
{
    type CudaRepresentation = WrapperCudaRepresentation<T>;
    type CudaAllocation = rc::common::CombinedCudaAlloc<
        <T as rc::common::RustToCuda>::CudaAllocation,
        rc::common::NullCudaAlloc,
    >;
    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow<CudaAllocType: rc::common::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
    ) -> rc::rustacuda::error::CudaResult<
        (
            rc::common::DeviceAccessible<Self::CudaRepresentation>,
            rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        ),
    > {
        let alloc_front = rc::common::NullCudaAlloc;
        let alloc_tail = alloc;
        let (field_inner_repr, alloc_front) = rc::common::RustToCuda::borrow(
            &self.inner,
            alloc_front,
        )?;
        let borrow = WrapperCudaRepresentation {
            inner: field_inner_repr,
        };
        Ok((
            rc::common::DeviceAccessible::from(borrow),
            rc::common::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }
    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore<CudaAllocType: rc::common::CudaAlloc>(
        &mut self,
        alloc: rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    ) -> rc::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        let alloc_front = rc::common::RustToCuda::restore(&mut self.inner, alloc_front)?;
        Ok(alloc_tail)
    }
}
unsafe impl<T> rc::common::RustToCudaAsync for Wrapper<T>
where
    T: rc::common::RustToCudaAsync,
{
    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow_async<CudaAllocType: rc::common::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
        stream: &rc::rustacuda::stream::Stream,
    ) -> rc::rustacuda::error::CudaResult<
        (
            rc::common::DeviceAccessible<Self::CudaRepresentation>,
            rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        ),
    > {
        let alloc_front = rc::common::NullCudaAlloc;
        let alloc_tail = alloc;
        let (field_inner_repr, alloc_front) = rc::common::RustToCudaAsync::borrow_async(
            &self.inner,
            alloc_front,
            stream,
        )?;
        let borrow = WrapperCudaRepresentation {
            inner: field_inner_repr,
        };
        Ok((
            rc::common::DeviceAccessible::from(borrow),
            rc::common::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }
    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore_async<CudaAllocType: rc::common::CudaAlloc>(
        &mut self,
        alloc: rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        stream: &rc::rustacuda::stream::Stream,
    ) -> rc::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        let alloc_front = rc::common::RustToCudaAsync::restore_async(
            &mut self.inner,
            alloc_front,
            stream,
        )?;
        Ok(alloc_tail)
    }
}
unsafe impl<T> rc::common::CudaAsRust for WrapperCudaRepresentation<T>
where
    T: rc::common::RustToCuda,
{
    type RustRepresentation = Wrapper<T>;
}
#[cuda(crate = "rc")]
pub struct Empty([u8; 0]);
#[allow(dead_code)]
#[doc(hidden)]
#[repr(C)]
#[layout(crate = "rc :: const_type_layout")]
pub struct EmptyCudaRepresentation(
    rc::common::DeviceAccessible<rc::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>,
);
unsafe impl const rc::const_type_layout::TypeLayout for EmptyCudaRepresentation {
    const TYPE_LAYOUT: rc::const_type_layout::TypeLayoutInfo<'static> = {
        rc::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: rc::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: &[
                    rc::const_type_layout::Field {
                        name: "0",
                        offset: {
                            {
                                #[allow(clippy::unneeded_field_pattern)]
                                let EmptyCudaRepresentation {
                                    0: _,
                                    ..
                                }: EmptyCudaRepresentation;
                                if let ::const_type_layout::MaybeUninhabited::Inhabited(
                                    uninit,
                                )
                                    = unsafe {
                                        <EmptyCudaRepresentation as ::const_type_layout::TypeLayout>::uninit()
                                    } {
                                    let base_ptr: *const EmptyCudaRepresentation = (&raw const uninit)
                                        .cast();
                                    #[allow(unused_unsafe)]
                                    let field_ptr = unsafe { &raw const (*base_ptr).0 };
                                    #[allow(clippy::cast_sign_loss)]
                                    let offset = unsafe {
                                        field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                                    };
                                    #[allow(clippy::forget_non_drop, clippy::forget_copy)]
                                    core::mem::forget(uninit);
                                    ::const_type_layout::MaybeUninhabited::Inhabited(offset)
                                } else {
                                    ::const_type_layout::MaybeUninhabited::Uninhabited
                                }
                            }
                        },
                        ty: ::core::any::type_name::<
                            rc::common::DeviceAccessible<
                                rc::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>,
                            >,
                        >(),
                    },
                ],
            },
        }
    };
    unsafe fn uninit() -> rc::const_type_layout::MaybeUninhabited<
        ::core::mem::MaybeUninit<Self>,
    > {
        if let (rc::const_type_layout::MaybeUninhabited::Inhabited(f_0))
            = (<rc::common::DeviceAccessible<
                rc::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>,
            > as rc::const_type_layout::TypeLayout>::uninit()) {
            rc::const_type_layout::MaybeUninhabited::Inhabited(
                ::core::mem::MaybeUninit::new(EmptyCudaRepresentation(f_0.assume_init())),
            )
        } else {
            rc::const_type_layout::MaybeUninhabited::Uninhabited
        }
    }
}
unsafe impl const rc::const_type_layout::TypeGraph for EmptyCudaRepresentation {
    fn populate_graph(graph: &mut rc::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as rc::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <rc::common::DeviceAccessible<
                rc::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>,
            > as rc::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
unsafe impl rc::rustacuda_core::DeviceCopy for EmptyCudaRepresentation {}
unsafe impl rc::common::RustToCuda for Empty {
    type CudaRepresentation = EmptyCudaRepresentation;
    type CudaAllocation = rc::common::NullCudaAlloc;
    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow<CudaAllocType: rc::common::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
    ) -> rc::rustacuda::error::CudaResult<
        (
            rc::common::DeviceAccessible<Self::CudaRepresentation>,
            rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        ),
    > {
        let alloc_front = rc::common::NullCudaAlloc;
        let alloc_tail = alloc;
        let field_0_repr = rc::common::DeviceAccessible::from(&self.0);
        let borrow = EmptyCudaRepresentation(field_0_repr);
        Ok((
            rc::common::DeviceAccessible::from(borrow),
            rc::common::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }
    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore<CudaAllocType: rc::common::CudaAlloc>(
        &mut self,
        alloc: rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    ) -> rc::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        Ok(alloc_tail)
    }
}
unsafe impl rc::common::RustToCudaAsync for Empty {
    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow_async<CudaAllocType: rc::common::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
        stream: &rc::rustacuda::stream::Stream,
    ) -> rc::rustacuda::error::CudaResult<
        (
            rc::common::DeviceAccessible<Self::CudaRepresentation>,
            rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        ),
    > {
        let alloc_front = rc::common::NullCudaAlloc;
        let alloc_tail = alloc;
        let field_0_repr = rc::common::DeviceAccessible::from(&self.0);
        let borrow = EmptyCudaRepresentation(field_0_repr);
        Ok((
            rc::common::DeviceAccessible::from(borrow),
            rc::common::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }
    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore_async<CudaAllocType: rc::common::CudaAlloc>(
        &mut self,
        alloc: rc::common::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
        stream: &rc::rustacuda::stream::Stream,
    ) -> rc::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        Ok(alloc_tail)
    }
}
unsafe impl rc::common::CudaAsRust for EmptyCudaRepresentation {
    type RustRepresentation = Empty;
}
#[repr(C)]
#[layout(crate = "rc::const_type_layout")]
pub struct Tuple(u32, i32);
unsafe impl const rc::const_type_layout::TypeLayout for Tuple {
    const TYPE_LAYOUT: rc::const_type_layout::TypeLayoutInfo<'static> = {
        rc::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: rc::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: &[
                    rc::const_type_layout::Field {
                        name: "0",
                        offset: {
                            {
                                #[allow(clippy::unneeded_field_pattern)]
                                let Tuple { 0: _, .. }: Tuple;
                                if let ::const_type_layout::MaybeUninhabited::Inhabited(
                                    uninit,
                                )
                                    = unsafe {
                                        <Tuple as ::const_type_layout::TypeLayout>::uninit()
                                    } {
                                    let base_ptr: *const Tuple = (&raw const uninit).cast();
                                    #[allow(unused_unsafe)]
                                    let field_ptr = unsafe { &raw const (*base_ptr).0 };
                                    #[allow(clippy::cast_sign_loss)]
                                    let offset = unsafe {
                                        field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                                    };
                                    #[allow(clippy::forget_non_drop, clippy::forget_copy)]
                                    core::mem::forget(uninit);
                                    ::const_type_layout::MaybeUninhabited::Inhabited(offset)
                                } else {
                                    ::const_type_layout::MaybeUninhabited::Uninhabited
                                }
                            }
                        },
                        ty: ::core::any::type_name::<u32>(),
                    },
                    rc::const_type_layout::Field {
                        name: "1",
                        offset: {
                            {
                                #[allow(clippy::unneeded_field_pattern)]
                                let Tuple { 1: _, .. }: Tuple;
                                if let ::const_type_layout::MaybeUninhabited::Inhabited(
                                    uninit,
                                )
                                    = unsafe {
                                        <Tuple as ::const_type_layout::TypeLayout>::uninit()
                                    } {
                                    let base_ptr: *const Tuple = (&raw const uninit).cast();
                                    #[allow(unused_unsafe)]
                                    let field_ptr = unsafe { &raw const (*base_ptr).1 };
                                    #[allow(clippy::cast_sign_loss)]
                                    let offset = unsafe {
                                        field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                                    };
                                    #[allow(clippy::forget_non_drop, clippy::forget_copy)]
                                    core::mem::forget(uninit);
                                    ::const_type_layout::MaybeUninhabited::Inhabited(offset)
                                } else {
                                    ::const_type_layout::MaybeUninhabited::Uninhabited
                                }
                            }
                        },
                        ty: ::core::any::type_name::<i32>(),
                    },
                ],
            },
        }
    };
    unsafe fn uninit() -> rc::const_type_layout::MaybeUninhabited<
        ::core::mem::MaybeUninit<Self>,
    > {
        if let (
            rc::const_type_layout::MaybeUninhabited::Inhabited(f_0),
            rc::const_type_layout::MaybeUninhabited::Inhabited(f_1),
        )
            = (
                <u32 as rc::const_type_layout::TypeLayout>::uninit(),
                <i32 as rc::const_type_layout::TypeLayout>::uninit(),
            ) {
            rc::const_type_layout::MaybeUninhabited::Inhabited(
                ::core::mem::MaybeUninit::new(
                    Tuple(f_0.assume_init(), f_1.assume_init()),
                ),
            )
        } else {
            rc::const_type_layout::MaybeUninhabited::Uninhabited
        }
    }
}
unsafe impl const rc::const_type_layout::TypeGraph for Tuple {
    fn populate_graph(graph: &mut rc::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as rc::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <u32 as rc::const_type_layout::TypeGraph>::populate_graph(graph);
            <i32 as rc::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
#[cfg(not(target_os = "cuda"))]
#[allow(clippy::missing_safety_doc)]
unsafe trait KernelArgs<T: rc::common::RustToCuda>
where
    T: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{
    type __T_0;
    type __T_1;
    type __T_2;
    type __T_3;
    type __T_4;
    type __T_5;
}
unsafe impl<T: rc::common::RustToCuda> KernelArgs<T> for ()
where
    T: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{
    type __T_0 = Dummy;
    type __T_1 = Wrapper<T>;
    type __T_2 = Wrapper<T>;
    type __T_3 = core::sync::atomic::AtomicU64;
    type __T_4 = Wrapper<T>;
    type __T_5 = Tuple;
}
#[cfg(not(target_os = "cuda"))]
#[allow(clippy::missing_safety_doc)]
unsafe trait KernelPtx<T: rc::common::RustToCuda>
where
    T: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{
    fn get_ptx_str() -> &'static str
    where
        Self: Sized + rc::host::Launcher<KernelTraitObject = dyn Kernel<T>>;
    fn new_kernel() -> rc::rustacuda::error::CudaResult<
        rc::host::TypedKernel<dyn Kernel<T>>,
    >
    where
        Self: Sized + rc::host::Launcher<KernelTraitObject = dyn Kernel<T>>;
}
#[cfg(not(target_os = "cuda"))]
#[allow(clippy::missing_safety_doc)]
unsafe trait Kernel<T: rc::common::RustToCuda>: KernelPtx<T>
where
    T: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{
    #[allow(clippy::needless_lifetimes)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::used_underscore_binding)]
    #[allow(unused_variables)]
    fn kernel<'stream, '__r2c_lt_0, '__r2c_lt_1, '__r2c_lt_2, '__r2c_move_lt_4, 'a>(
        &mut self,
        stream: &'stream rc::rustacuda::stream::Stream,
        _x: &'__r2c_lt_0 <() as KernelArgs<T>>::__T_0,
        _y: &'__r2c_lt_1 mut <() as KernelArgs<T>>::__T_1,
        _z: &'__r2c_lt_2 <() as KernelArgs<T>>::__T_2,
        _v: &'a <() as KernelArgs<T>>::__T_3,
        kernel_arg_4: <() as KernelArgs<T>>::__T_4,
        s_t: <() as KernelArgs<T>>::__T_5,
    ) -> rc::rustacuda::error::CudaResult<()>
    where
        Self: Sized + rc::host::Launcher<KernelTraitObject = dyn Kernel<T>>,
    {
        const fn __check_is_sync<T: ?Sized>(_x: &T) -> bool {
            trait IsSyncMarker {
                const SYNC: bool = false;
            }
            impl<T: ?Sized> IsSyncMarker for T {}
            struct CheckIs<T: ?Sized>(::core::marker::PhantomData<T>);
            #[allow(dead_code)]
            impl<T: ?Sized + Sync> CheckIs<T> {
                const SYNC: bool = true;
            }
            <CheckIs<T>>::SYNC
        }
        let mut ___x_box = rc::host::HostDeviceBox::from(
            rc::rustacuda::memory::DeviceBox::new(
                rc::utils::device_copy::SafeDeviceCopyWrapper::from_ref(_x),
            )?,
        );
        #[allow(clippy::redundant_closure_call)]
        let __result = (|_x| {
            rc::host::LendToCuda::lend_to_cuda_mut(
                _y,
                |mut _y| {
                    (|_y| {
                        rc::host::LendToCuda::lend_to_cuda(
                            _z,
                            |_z| {
                                (|_z| {
                                    let mut ___v_box = rc::host::HostDeviceBox::from(
                                        rc::rustacuda::memory::DeviceBox::new(
                                            rc::utils::device_copy::SafeDeviceCopyWrapper::from_ref(_v),
                                        )?,
                                    );
                                    #[allow(clippy::redundant_closure_call)]
                                    let __result = (|_v| {
                                        rc::host::LendToCuda::move_to_cuda(
                                            kernel_arg_4,
                                            |mut kernel_arg_4| {
                                                (|kernel_arg_4| {
                                                    {
                                                        let s_t = rc::utils::device_copy::SafeDeviceCopyWrapper::from(
                                                            s_t,
                                                        );
                                                        self.kernel_async(
                                                            stream,
                                                            _x,
                                                            _y,
                                                            _z,
                                                            _v,
                                                            kernel_arg_4,
                                                            s_t,
                                                        )?;
                                                        stream.synchronize()
                                                    }
                                                })(kernel_arg_4.as_async())
                                            },
                                        )
                                    })(unsafe {
                                        rc::host::HostAndDeviceConstRef::new(
                                                &___v_box,
                                                rc::utils::device_copy::SafeDeviceCopyWrapper::from_ref(_v),
                                            )
                                            .as_async()
                                    });
                                    if !__check_is_sync(_v) {
                                        ___v_box
                                            .copy_to(unsafe { &mut *(_v as *const _ as *mut _) })?;
                                    }
                                    ::core::mem::drop(___v_box);
                                    __result
                                })(_z.as_async())
                            },
                        )
                    })(_y.as_async())
                },
            )
        })(unsafe {
            rc::host::HostAndDeviceConstRef::new(
                    &___x_box,
                    rc::utils::device_copy::SafeDeviceCopyWrapper::from_ref(_x),
                )
                .as_async()
        });
        if !__check_is_sync(_x) {
            ___x_box.copy_to(unsafe { &mut *(_x as *const _ as *mut _) })?;
        }
        ::core::mem::drop(___x_box);
        __result
    }
    #[allow(clippy::extra_unused_type_parameters)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::used_underscore_binding)]
    #[allow(unused_variables)]
    fn kernel_async<
        'stream,
        '__r2c_lt_0,
        '__r2c_lt_1,
        '__r2c_lt_2,
        '__r2c_move_lt_4,
        'a,
    >(
        &mut self,
        stream: &'stream rc::rustacuda::stream::Stream,
        _x: rc::host::HostAndDeviceConstRefAsync<
            'stream,
            '__r2c_lt_0,
            rc::utils::device_copy::SafeDeviceCopyWrapper<<() as KernelArgs<T>>::__T_0>,
        >,
        mut _y: rc::host::HostAndDeviceMutRefAsync<
            'stream,
            '__r2c_lt_1,
            rc::common::DeviceAccessible<
                <<() as KernelArgs<
                    T,
                >>::__T_1 as rc::common::RustToCuda>::CudaRepresentation,
            >,
        >,
        _z: rc::host::HostAndDeviceConstRefAsync<
            'stream,
            '__r2c_lt_2,
            rc::common::DeviceAccessible<
                <<() as KernelArgs<
                    T,
                >>::__T_2 as rc::common::RustToCuda>::CudaRepresentation,
            >,
        >,
        _v: rc::host::HostAndDeviceConstRefAsync<
            'stream,
            'a,
            rc::utils::device_copy::SafeDeviceCopyWrapper<<() as KernelArgs<T>>::__T_3>,
        >,
        kernel_arg_4: rc::host::HostAndDeviceOwnedAsync<
            'stream,
            '__r2c_move_lt_4,
            rc::common::DeviceAccessible<
                <<() as KernelArgs<
                    T,
                >>::__T_4 as rc::common::RustToCuda>::CudaRepresentation,
            >,
        >,
        s_t: rc::utils::device_copy::SafeDeviceCopyWrapper<<() as KernelArgs<T>>::__T_5>,
    ) -> rc::rustacuda::error::CudaResult<()>
    where
        Self: Sized + rc::host::Launcher<KernelTraitObject = dyn Kernel<T>>,
    {
        let rc::host::LaunchPackage { kernel, watcher, config } = rc::host::Launcher::get_launch_package(
            self,
        );
        let kernel_jit_result = if config.ptx_jit {
            kernel
                .compile_with_ptx_jit_args(
                    Some(
                        &[
                            None,
                            Some(rc::ptx_jit::arg_as_raw_bytes(_y.for_host())),
                            None,
                            Some(rc::ptx_jit::arg_as_raw_bytes(_v.for_host())),
                            None,
                            None,
                        ],
                    ),
                )?
        } else {
            kernel.compile_with_ptx_jit_args(None)?
        };
        let function = match kernel_jit_result {
            rc::host::KernelJITResult::Recompiled(function) => {
                <Self as rc::host::Launcher>::on_compile(function, watcher)?;
                function
            }
            rc::host::KernelJITResult::Cached(function) => function,
        };
        #[allow(clippy::redundant_closure_call)]
        (|
            _x: rc::common::DeviceConstRef<
                '__r2c_lt_0,
                rc::utils::device_copy::SafeDeviceCopyWrapper<
                    <() as KernelArgs<T>>::__T_0,
                >,
            >,
            _y: rc::common::DeviceMutRef<
                '__r2c_lt_1,
                rc::common::DeviceAccessible<
                    <<() as KernelArgs<
                        T,
                    >>::__T_1 as rc::common::RustToCuda>::CudaRepresentation,
                >,
            >,
            _z: rc::common::DeviceConstRef<
                '__r2c_lt_2,
                rc::common::DeviceAccessible<
                    <<() as KernelArgs<
                        T,
                    >>::__T_2 as rc::common::RustToCuda>::CudaRepresentation,
                >,
            >,
            _v: rc::common::DeviceConstRef<
                'a,
                rc::utils::device_copy::SafeDeviceCopyWrapper<
                    <() as KernelArgs<T>>::__T_3,
                >,
            >,
            kernel_arg_4: rc::common::DeviceMutRef<
                '__r2c_move_lt_4,
                rc::common::DeviceAccessible<
                    <<() as KernelArgs<
                        T,
                    >>::__T_4 as rc::common::RustToCuda>::CudaRepresentation,
                >,
            >,
            s_t: rc::utils::device_copy::SafeDeviceCopyWrapper<
                <() as KernelArgs<T>>::__T_5,
            >|
        {
            if false {
                #[allow(dead_code)]
                fn assert_impl_devicecopy<T: rc::rustacuda_core::DeviceCopy>(_val: &T) {}
                #[allow(dead_code)]
                fn assert_impl_no_aliasing<T: rc::safety::NoAliasing>() {}
                #[allow(dead_code)]
                fn assert_impl_fits_into_device_register<
                    T: rc::safety::FitsIntoDeviceRegister,
                >(_val: &T) {}
                assert_impl_devicecopy(&_x);
                assert_impl_devicecopy(&_y);
                assert_impl_devicecopy(&_z);
                assert_impl_devicecopy(&_v);
                assert_impl_devicecopy(&kernel_arg_4);
                assert_impl_devicecopy(&s_t);
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_0>();
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_1>();
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_2>();
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_3>();
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_4>();
                assert_impl_no_aliasing::<<() as KernelArgs<T>>::__T_5>();
                assert_impl_fits_into_device_register(&_x);
                assert_impl_fits_into_device_register(&_y);
                assert_impl_fits_into_device_register(&_z);
                assert_impl_fits_into_device_register(&_v);
                assert_impl_fits_into_device_register(&kernel_arg_4);
                assert_impl_fits_into_device_register(&s_t);
            }
            let rc::host::LaunchConfig { grid, block, shared_memory_size, ptx_jit: _ } = config;
            unsafe {
                stream
                    .launch(
                        function,
                        grid,
                        block,
                        shared_memory_size,
                        &[
                            &_x as *const _ as *mut ::std::ffi::c_void,
                            &_y as *const _ as *mut ::std::ffi::c_void,
                            &_z as *const _ as *mut ::std::ffi::c_void,
                            &_v as *const _ as *mut ::std::ffi::c_void,
                            &kernel_arg_4 as *const _ as *mut ::std::ffi::c_void,
                            &s_t as *const _ as *mut ::std::ffi::c_void,
                        ],
                    )
            }
        })(
            unsafe { _x.for_device_async() },
            unsafe { _y.for_device_async() },
            unsafe { _z.for_device_async() },
            unsafe { _v.for_device_async() },
            unsafe { kernel_arg_4.for_device_async() },
            s_t,
        )
    }
}
#[cfg(not(target_os = "cuda"))]
#[allow(clippy::missing_safety_doc)]
unsafe impl<T: rc::common::RustToCuda, K: KernelPtx<T>> Kernel<T> for K
where
    T: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{}
#[cfg(not(target_os = "cuda"))]
const _: rc::safety::kernel_signature::Assert<
    { rc::safety::kernel_signature::CpuAndGpuKernelSignatures::Match },
> = rc::safety::kernel_signature::Assert::<
    {
        rc::safety::kernel_signature::check(
            "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 3.2\n.target sm_35\n.address_size 64\n\n\t// .globl\tkernel_type_layout\n\n.visible .entry kernel_type_layout()\n{\n\n\n\tret;\n\n}\n\t// .globl\tkernel_dfae7eaf723a670c\n.visible .entry kernel_dfae7eaf723a670c()\n{\n\n\n\tret;\n\n}\n"
                .as_bytes(),
            ".visible .entry kernel_dfae7eaf723a670c".as_bytes(),
        )
    },
>;
#[cfg(not(target_os = "cuda"))]
mod host {
    #[allow(unused_imports)]
    use super::KernelArgs;
    use super::{Kernel, KernelPtx};
    #[allow(dead_code)]
    struct Launcher<T: rc::common::RustToCuda>(core::marker::PhantomData<T>);
    unsafe impl KernelPtx<crate::Empty> for Launcher<crate::Empty> {
        fn get_ptx_str() -> &'static str {
            const PTX_STR: &'static str = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 3.2\n.target sm_35\n.address_size 64\n\n\t// .globl\tkernel_dfae7eaf723a670c_kernel_aab1c403129e575b\n.visible .entry kernel_dfae7eaf723a670c_kernel_aab1c403129e575b(\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_0,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_1,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_2,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_3,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_4,\n\t.param .align 4 .b8 kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_5[8]\n)\n{\n\t.reg .b32 \t%r<6>;\n\t.reg .b64 \t%rd<7>;\n\t.reg .f64 \t%fd<5>;\n\n\tld.param.u64 \t%rd3, [kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_3];\n\tcvta.to.global.u64 \t%rd4, %rd3;\n\tld.param.u64 \t%rd5, [kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_1];\n\tcvta.to.global.u64 \t%rd6, %rd5;\n\tld.global.u32 \t%r1, [%rd6];\n\t// begin inline asm\n\t// <rust-cuda-ptx-jit-const-load-%r1-1> //\n\t// end inline asm\n\tld.param.u32 \t%r3, [kernel_dfae7eaf723a670c_kernel_aab1c403129e575b_param_5];\n\tld.global.u32 \t%r2, [%rd4];\n\t// begin inline asm\n\t// <rust-cuda-ptx-jit-const-load-%r2-3> //\n\t// end inline asm\n\t// begin inline asm\n\t.shared .align 4 .b8 %rd1_rust_cuda_static_shared[24];\ncvta.shared.u64 %rd1, %rd1_rust_cuda_static_shared;\n\t// end inline asm\n\t// begin inline asm\n\t.shared .align 4 .b8 %rd2_rust_cuda_static_shared[24];\ncvta.shared.u64 %rd2, %rd2_rust_cuda_static_shared;\n\t// end inline asm\n\tcvt.rn.f64.u32 \t%fd1, %r3;\n\tadd.rn.f64 \t%fd2, %fd1, %fd1;\n\tmax.f64 \t%fd3, %fd2, 0d0000000000000000;\n\tmin.f64 \t%fd4, %fd3, 0d41EFFFFFFFE00000;\n\tcvt.rzi.u32.f64 \t%r4, %fd4;\n\tst.u32 \t[%rd1+8], %r4;\n\tmov.u32 \t%r5, 24;\n\tst.u32 \t[%rd2+20], %r5;\n\tret;\n\n}\n\n// <crate::Empty>\n";
            const __KERNEL_DFAE7EAF723A670C__X_LAYOUT: &[u8; 879usize] = b"\xef\x06\x050.1.0mrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x06mrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x08\x08s\x0btransparent\x02\x07pointerh\x00Q*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\treferenceh\x00fcore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>Q*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\x08\x08pJrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>iJrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\x04\x04s\x0btransparent\x01\x010h\x00\x14single_source::Dummy\x14single_source::Dummy\x04\x04s\x01C\x01\x010h\x00\x03i32\x03i32\x04\x04vfcore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__Y_LAYOUT: &[u8; 1811usize] = b"\x93\x0e\x050.1.0\x84\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x0b\x84\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00h*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\treferenceh\x00\x83\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>h*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x08\x08pcrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>mcrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x00\x01s\x0btransparent\x01\x010h\x00>single_source::WrapperCudaRepresentation<single_source::Empty>>single_source::WrapperCudaRepresentation<single_source::Empty>\x00\x01s\x01C\x01\x05innerh\x00Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>\x00\x01s\x0btransparent\x01\x010h\x00&single_source::EmptyCudaRepresentation&single_source::EmptyCudaRepresentation\x00\x01s\x01C\x01\x010h\x00brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>\x00\x01s\x0btransparent\x01\x010h\x00=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>\x00\x01s\x0btransparent\x01\x010h\x00\x07[u8; 0]\x07[u8; 0]\x00\x01a\x02u8\x00\x02u8\x01\x01v\x83\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__Z_LAYOUT: &[u8; 1809usize] = b"\x91\x0e\x050.1.0\x86\x01rust_cuda::common::DeviceConstRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x0b\x86\x01rust_cuda::common::DeviceConstRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00j*const rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\treferenceh\x00\x7fcore::marker::PhantomData<&rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>j*const rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x08\x08pcrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>icrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x00\x01s\x0btransparent\x01\x010h\x00>single_source::WrapperCudaRepresentation<single_source::Empty>>single_source::WrapperCudaRepresentation<single_source::Empty>\x00\x01s\x01C\x01\x05innerh\x00Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>\x00\x01s\x0btransparent\x01\x010h\x00&single_source::EmptyCudaRepresentation&single_source::EmptyCudaRepresentation\x00\x01s\x01C\x01\x010h\x00brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>\x00\x01s\x0btransparent\x01\x010h\x00=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>\x00\x01s\x0btransparent\x01\x010h\x00\x07[u8; 0]\x07[u8; 0]\x00\x01a\x02u8\x00\x02u8\x01\x01v\x7fcore::marker::PhantomData<&rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__V_LAYOUT: &[u8; 1068usize] = b"\xac\x08\x050.1.0vrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x07vrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x08\x08s\x0btransparent\x02\x07pointerh\x00Z*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\treferenceh\x00ocore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>Z*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\x08\x08pSrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>iSrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\x08\x08s\x0btransparent\x01\x010h\x00\x1dcore::sync::atomic::AtomicU64\x1dcore::sync::atomic::AtomicU64\x08\x08s\nC,align(8)\x01\x01vh\x00\x1bcore::cell::UnsafeCell<u64>\x1bcore::cell::UnsafeCell<u64>\x08\x08s\x15no_nieche,transparent\x01\x05valueh\x00\x03u64\x03u64\x08\x08vocore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C_KERNEL_ARG_4_LAYOUT: &[u8; 1811usize] = b"\x93\x0e\x050.1.0\x84\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x0b\x84\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00h*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\treferenceh\x00\x83\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>h*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x08\x08pcrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>mcrust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>\x00\x01s\x0btransparent\x01\x010h\x00>single_source::WrapperCudaRepresentation<single_source::Empty>>single_source::WrapperCudaRepresentation<single_source::Empty>\x00\x01s\x01C\x01\x05innerh\x00Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>Krust_cuda::common::DeviceAccessible<single_source::EmptyCudaRepresentation>\x00\x01s\x0btransparent\x01\x010h\x00&single_source::EmptyCudaRepresentation&single_source::EmptyCudaRepresentation\x00\x01s\x01C\x01\x010h\x00brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>brust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>>\x00\x01s\x0btransparent\x01\x010h\x00=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>=rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<[u8; 0]>\x00\x01s\x0btransparent\x01\x010h\x00\x07[u8; 0]\x07[u8; 0]\x00\x01a\x02u8\x00\x02u8\x01\x01v\x83\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<single_source::Empty>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C_S_T_LAYOUT: &[u8; 257usize] = b"\x81\x02\x050.1.0Jrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Tuple>\x04Jrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Tuple>\x08\x04s\x0btransparent\x01\x010h\x00\x14single_source::Tuple\x14single_source::Tuple\x08\x04s\x01C\x02\x010h\x00\x03u32\x011h\x04\x03i32\x03u32\x04\x04v\x03i32\x04\x04v";
            const _: rc::safety::kernel_signature::Assert<
                { rc::safety::kernel_signature::CpuAndGpuKernelSignatures::Match },
            > = rc::safety::kernel_signature::Assert::<
                {
                    rc::safety::kernel_signature::check(
                        PTX_STR.as_bytes(),
                        ".visible .entry kernel_dfae7eaf723a670c_kernel_aab1c403129e575b"
                            .as_bytes(),
                    )
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::utils::device_copy::SafeDeviceCopyWrapper<
                                <() as KernelArgs<crate::Empty>>::__T_0,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__X_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceMutRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    crate::Empty,
                                >>::__T_1 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__Y_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    crate::Empty,
                                >>::__T_2 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__Z_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::utils::device_copy::SafeDeviceCopyWrapper<
                                <() as KernelArgs<crate::Empty>>::__T_3,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__V_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceMutRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    crate::Empty,
                                >>::__T_4 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C_KERNEL_ARG_4_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::utils::device_copy::SafeDeviceCopyWrapper<
                            <() as KernelArgs<crate::Empty>>::__T_5,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C_S_T_LAYOUT)
                },
            >;
            PTX_STR
        }
        fn new_kernel() -> rc::rustacuda::error::CudaResult<
            rc::host::TypedKernel<dyn Kernel<crate::Empty>>,
        > {
            let ptx = Self::get_ptx_str();
            let entry_point = "kernel_dfae7eaf723a670c_kernel_aab1c403129e575b";
            rc::host::TypedKernel::new(ptx, entry_point)
        }
    }
    unsafe impl KernelPtx<rc::utils::device_copy::SafeDeviceCopyWrapper<u64>>
    for Launcher<rc::utils::device_copy::SafeDeviceCopyWrapper<u64>> {
        fn get_ptx_str() -> &'static str {
            const PTX_STR: &'static str = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 3.2\n.target sm_35\n.address_size 64\n\n\t// .globl\tkernel_dfae7eaf723a670c_kernel_54d0891c50855d77\n.visible .entry kernel_dfae7eaf723a670c_kernel_54d0891c50855d77(\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_0,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_1,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_2,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_3,\n\t.param .u64 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_4,\n\t.param .align 4 .b8 kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_5[8]\n)\n{\n\t.reg .b32 \t%r<6>;\n\t.reg .b64 \t%rd<7>;\n\t.reg .f64 \t%fd<5>;\n\n\tld.param.u64 \t%rd3, [kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_3];\n\tcvta.to.global.u64 \t%rd4, %rd3;\n\tld.param.u64 \t%rd5, [kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_1];\n\tcvta.to.global.u64 \t%rd6, %rd5;\n\tld.global.u32 \t%r1, [%rd6];\n\t// begin inline asm\n\t// <rust-cuda-ptx-jit-const-load-%r1-1> //\n\t// end inline asm\n\tld.param.u32 \t%r3, [kernel_dfae7eaf723a670c_kernel_54d0891c50855d77_param_5];\n\tld.global.u32 \t%r2, [%rd4];\n\t// begin inline asm\n\t// <rust-cuda-ptx-jit-const-load-%r2-3> //\n\t// end inline asm\n\t// begin inline asm\n\t.shared .align 4 .b8 %rd1_rust_cuda_static_shared[24];\ncvta.shared.u64 %rd1, %rd1_rust_cuda_static_shared;\n\t// end inline asm\n\t// begin inline asm\n\t.shared .align 4 .b8 %rd2_rust_cuda_static_shared[24];\ncvta.shared.u64 %rd2, %rd2_rust_cuda_static_shared;\n\t// end inline asm\n\tcvt.rn.f64.u32 \t%fd1, %r3;\n\tadd.rn.f64 \t%fd2, %fd1, %fd1;\n\tmax.f64 \t%fd3, %fd2, 0d0000000000000000;\n\tmin.f64 \t%fd4, %fd3, 0d41EFFFFFFFE00000;\n\tcvt.rzi.u32.f64 \t%r4, %fd4;\n\tst.u32 \t[%rd1+8], %r4;\n\tmov.u32 \t%r5, 24;\n\tst.u32 \t[%rd2+20], %r5;\n\tret;\n\n}\n\n// <rc::utils::device_copy::SafeDeviceCopyWrapper<u64>>\n";
            const __KERNEL_DFAE7EAF723A670C__X_LAYOUT: &[u8; 879usize] = b"\xef\x06\x050.1.0mrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x06mrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x08\x08s\x0btransparent\x02\x07pointerh\x00Q*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\treferenceh\x00fcore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>Q*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\x08\x08pJrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>iJrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>\x04\x04s\x0btransparent\x01\x010h\x00\x14single_source::Dummy\x14single_source::Dummy\x04\x04s\x01C\x01\x010h\x00\x03i32\x03i32\x04\x04vfcore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Dummy>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__Y_LAYOUT: &[u8; 1891usize] = b"\xe3\x0e\x050.1.0\xa9\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\xa9\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00\x8d\x01*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\treferenceh\x00\xa8\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x8d\x01*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08p\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>m\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08s\x0btransparent\x01\x010h\x00csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x01C\x01\x05innerh\x00^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x0btransparent\x01\x010h\x009rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>9rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>\x08\x08s\x0btransparent\x01\x010h\x00\x03u64\x03u64\x08\x08v\xa8\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__Z_LAYOUT: &[u8; 1891usize] = b"\xe3\x0e\x050.1.0\xab\x01rust_cuda::common::DeviceConstRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\xab\x01rust_cuda::common::DeviceConstRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00\x8f\x01*const rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\treferenceh\x00\xa4\x01core::marker::PhantomData<&rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x8f\x01*const rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08p\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>i\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08s\x0btransparent\x01\x010h\x00csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x01C\x01\x05innerh\x00^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x0btransparent\x01\x010h\x009rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>9rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>\x08\x08s\x0btransparent\x01\x010h\x00\x03u64\x03u64\x08\x08v\xa4\x01core::marker::PhantomData<&rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C__V_LAYOUT: &[u8; 1068usize] = b"\xac\x08\x050.1.0vrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x07vrust_cuda::common::DeviceConstRef<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x08\x08s\x0btransparent\x02\x07pointerh\x00Z*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\treferenceh\x00ocore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>Z*const rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\x08\x08pSrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>iSrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>\x08\x08s\x0btransparent\x01\x010h\x00\x1dcore::sync::atomic::AtomicU64\x1dcore::sync::atomic::AtomicU64\x08\x08s\nC,align(8)\x01\x01vh\x00\x1bcore::cell::UnsafeCell<u64>\x1bcore::cell::UnsafeCell<u64>\x08\x08s\x15no_nieche,transparent\x01\x05valueh\x00\x03u64\x03u64\x08\x08vocore::marker::PhantomData<&rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<core::sync::atomic::AtomicU64>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C_KERNEL_ARG_4_LAYOUT: &[u8; 1891usize] = b"\xe3\x0e\x050.1.0\xa9\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\xa9\x01rust_cuda::common::DeviceMutRef<rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x08\x08s\x0btransparent\x02\x07pointerh\x00\x8d\x01*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\treferenceh\x00\xa8\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x8d\x01*mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08p\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>m\x88\x01rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>\x08\x08s\x0btransparent\x01\x010h\x00csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>csingle_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x01C\x01\x05innerh\x00^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>^rust_cuda::common::DeviceAccessible<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>\x08\x08s\x0btransparent\x01\x010h\x009rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>9rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>\x08\x08s\x0btransparent\x01\x010h\x00\x03u64\x03u64\x08\x08v\xa8\x01core::marker::PhantomData<&mut rust_cuda::common::DeviceAccessible<single_source::WrapperCudaRepresentation<rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>>>>\x00\x01s\x00\x00";
            const __KERNEL_DFAE7EAF723A670C_S_T_LAYOUT: &[u8; 257usize] = b"\x81\x02\x050.1.0Jrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Tuple>\x04Jrust_cuda::utils::device_copy::SafeDeviceCopyWrapper<single_source::Tuple>\x08\x04s\x0btransparent\x01\x010h\x00\x14single_source::Tuple\x14single_source::Tuple\x08\x04s\x01C\x02\x010h\x00\x03u32\x011h\x04\x03i32\x03u32\x04\x04v\x03i32\x04\x04v";
            const _: rc::safety::kernel_signature::Assert<
                { rc::safety::kernel_signature::CpuAndGpuKernelSignatures::Match },
            > = rc::safety::kernel_signature::Assert::<
                {
                    rc::safety::kernel_signature::check(
                        PTX_STR.as_bytes(),
                        ".visible .entry kernel_dfae7eaf723a670c_kernel_54d0891c50855d77"
                            .as_bytes(),
                    )
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::utils::device_copy::SafeDeviceCopyWrapper<
                                <() as KernelArgs<
                                    rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                                >>::__T_0,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__X_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceMutRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                                >>::__T_1 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__Y_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                                >>::__T_2 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__Z_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceConstRef<
                            'static,
                            rc::utils::device_copy::SafeDeviceCopyWrapper<
                                <() as KernelArgs<
                                    rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                                >>::__T_3,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C__V_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::common::DeviceMutRef<
                            'static,
                            rc::common::DeviceAccessible<
                                <<() as KernelArgs<
                                    rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                                >>::__T_4 as rc::common::RustToCuda>::CudaRepresentation,
                            >,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C_KERNEL_ARG_4_LAYOUT)
                },
            >;
            const _: rc::safety::type_layout::Assert<
                { rc::safety::type_layout::CpuAndGpuTypeLayouts::Match },
            > = rc::safety::type_layout::Assert::<
                {
                    rc::safety::type_layout::check::<
                        rc::utils::device_copy::SafeDeviceCopyWrapper<
                            <() as KernelArgs<
                                rc::utils::device_copy::SafeDeviceCopyWrapper<u64>,
                            >>::__T_5,
                        >,
                    >(__KERNEL_DFAE7EAF723A670C_S_T_LAYOUT)
                },
            >;
            PTX_STR
        }
        fn new_kernel() -> rc::rustacuda::error::CudaResult<
            rc::host::TypedKernel<
                dyn Kernel<rc::utils::device_copy::SafeDeviceCopyWrapper<u64>>,
            >,
        > {
            let ptx = Self::get_ptx_str();
            let entry_point = "kernel_dfae7eaf723a670c_kernel_54d0891c50855d77";
            rc::host::TypedKernel::new(ptx, entry_point)
        }
    }
    impl<T: rc::common::RustToCuda> rc::host::Launcher for Launcher<T> {
        type CompilationWatcher = ();
        type KernelTraitObject = dyn Kernel<T>;
        fn get_launch_package(&mut self) -> rc::host::LaunchPackage<Self> {
            ::core::panicking::panic("not implemented")
        }
    }
}
