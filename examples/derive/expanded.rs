#![feature(prelude_import)]
#![deny(clippy::pedantic)]
#![feature(const_trait_impl)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
struct Inner<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[r2cEmbed]
    inner: T,
}
#[allow(dead_code)]
#[doc(hidden)]
#[repr(C)]
struct InnerCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    inner: rust_cuda::common::DeviceAccessible<
        <T as rust_cuda::common::RustToCuda>::CudaRepresentation,
    >,
}
unsafe impl<T> ::const_type_layout::TypeLayout for InnerCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<<T as rust_cuda::common::RustToCuda>::CudaRepresentation>:
        ::const_type_layout::TypeLayout,
{
    const TYPE_LAYOUT: ::const_type_layout::TypeLayoutInfo<'static> = {
        ::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: ::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: Self::__INNERCUDAREPRESENTATION_FIELDS,
            },
        }
    };
}
impl<T> InnerCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<<T as rust_cuda::common::RustToCuda>::CudaRepresentation>:
        ::const_type_layout::TypeLayout,
{
    const __INNERCUDAREPRESENTATION_FIELDS: &'static [::const_type_layout::Field<'static>; 1usize] =
        &[::const_type_layout::Field {
            name: "inner",
            offset: {
                let uninit = ::core::mem::MaybeUninit::<InnerCudaRepresentation<T>>::uninit();
                let base_ptr: *const InnerCudaRepresentation<T> = uninit.as_ptr();
                # [allow (clippy :: unneeded_field_pattern)] let InnerCudaRepresentation { inner : _ , .. } : InnerCudaRepresentation < T > ;
                #[allow(unused_unsafe)]
                let field_ptr = unsafe { &raw const (*base_ptr).inner };
                #[allow(clippy::cast_sign_loss)]
                unsafe {
                    field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                }
            },
            ty: ::core::any::type_name::<
                rust_cuda::common::DeviceAccessible<
                    <T as rust_cuda::common::RustToCuda>::CudaRepresentation,
                >,
            >(),
        }];
}
unsafe impl<T> const ::const_type_layout::TypeGraph for InnerCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<<T as rust_cuda::common::RustToCuda>::CudaRepresentation>:
        ::const_type_layout::TypeGraph,
{
    fn populate_graph(graph: &mut ::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as ::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <rust_cuda::common::DeviceAccessible<
                <T as rust_cuda::common::RustToCuda>::CudaRepresentation,
            > as ::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
unsafe impl<T> rust_cuda::rustacuda_core::DeviceCopy for InnerCudaRepresentation<T> where
    T: Copy + rust_cuda::common::RustToCuda
{
}
unsafe impl<T> rust_cuda::common::RustToCuda for Inner<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[cfg(not(target_os = "cuda"))]
    type CudaAllocation = rust_cuda::host::CombinedCudaAlloc<
        <T as rust_cuda::common::RustToCuda>::CudaAllocation,
        rust_cuda::host::NullCudaAlloc,
    >;
    type CudaRepresentation = InnerCudaRepresentation<T>;

    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow<CudaAllocType: rust_cuda::host::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
    ) -> rust_cuda::rustacuda::error::CudaResult<(
        rust_cuda::common::DeviceAccessible<Self::CudaRepresentation>,
        rust_cuda::host::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    )> {
        let alloc_front = rust_cuda::host::NullCudaAlloc;
        let alloc_tail = alloc;
        let (field_inner_repr, alloc_front) =
            rust_cuda::common::RustToCuda::borrow(&self.inner, alloc_front)?;
        let borrow = InnerCudaRepresentation {
            inner: field_inner_repr,
        };
        Ok((
            rust_cuda::common::DeviceAccessible::from(borrow),
            rust_cuda::host::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }

    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore<CudaAllocType: rust_cuda::host::CudaAlloc>(
        &mut self,
        alloc: rust_cuda::host::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    ) -> rust_cuda::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        let alloc_front = rust_cuda::common::RustToCuda::restore(&mut self.inner, alloc_front)?;
        Ok(alloc_tail)
    }
}
unsafe impl<T> rust_cuda::common::CudaAsRust for InnerCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    type RustRepresentation = Inner<T>;
}
struct Outer<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[r2cEmbed]
    inner: Inner<T>,
}
#[allow(dead_code)]
#[doc(hidden)]
#[repr(C)]
struct OuterCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    inner: rust_cuda::common::DeviceAccessible<
        <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
    >,
}
unsafe impl<T> ::const_type_layout::TypeLayout for OuterCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<
        <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
    >: ::const_type_layout::TypeLayout,
{
    const TYPE_LAYOUT: ::const_type_layout::TypeLayoutInfo<'static> = {
        ::const_type_layout::TypeLayoutInfo {
            name: ::core::any::type_name::<Self>(),
            size: ::core::mem::size_of::<Self>(),
            alignment: ::core::mem::align_of::<Self>(),
            structure: ::const_type_layout::TypeStructure::Struct {
                repr: "C",
                fields: Self::__OUTERCUDAREPRESENTATION_FIELDS,
            },
        }
    };
}
impl<T> OuterCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<
        <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
    >: ::const_type_layout::TypeLayout,
{
    const __OUTERCUDAREPRESENTATION_FIELDS: &'static [::const_type_layout::Field<'static>; 1usize] =
        &[::const_type_layout::Field {
            name: "inner",
            offset: {
                let uninit = ::core::mem::MaybeUninit::<OuterCudaRepresentation<T>>::uninit();
                let base_ptr: *const OuterCudaRepresentation<T> = uninit.as_ptr();
                # [allow (clippy :: unneeded_field_pattern)] let OuterCudaRepresentation { inner : _ , .. } : OuterCudaRepresentation < T > ;
                #[allow(unused_unsafe)]
                let field_ptr = unsafe { &raw const (*base_ptr).inner };
                #[allow(clippy::cast_sign_loss)]
                unsafe {
                    field_ptr.cast::<u8>().offset_from(base_ptr.cast()) as usize
                }
            },
            ty: ::core::any::type_name::<
                rust_cuda::common::DeviceAccessible<
                    <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
                >,
            >(),
        }];
}
unsafe impl<T> const ::const_type_layout::TypeGraph for OuterCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
    rust_cuda::common::DeviceAccessible<
        <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
    >: ::const_type_layout::TypeGraph,
{
    fn populate_graph(graph: &mut ::const_type_layout::TypeLayoutGraph<'static>) {
        if graph.insert(&<Self as ::const_type_layout::TypeLayout>::TYPE_LAYOUT) {
            <rust_cuda::common::DeviceAccessible<
                <Inner<T> as rust_cuda::common::RustToCuda>::CudaRepresentation,
            > as ::const_type_layout::TypeGraph>::populate_graph(graph);
        }
    }
}
unsafe impl<T> rust_cuda::rustacuda_core::DeviceCopy for OuterCudaRepresentation<T> where
    T: Copy + rust_cuda::common::RustToCuda
{
}
unsafe impl<T> rust_cuda::common::RustToCuda for Outer<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    #[cfg(not(target_os = "cuda"))]
    type CudaAllocation = rust_cuda::host::CombinedCudaAlloc<
        <Inner<T> as rust_cuda::common::RustToCuda>::CudaAllocation,
        rust_cuda::host::NullCudaAlloc,
    >;
    type CudaRepresentation = OuterCudaRepresentation<T>;

    #[cfg(not(target_os = "cuda"))]
    unsafe fn borrow<CudaAllocType: rust_cuda::host::CudaAlloc>(
        &self,
        alloc: CudaAllocType,
    ) -> rust_cuda::rustacuda::error::CudaResult<(
        rust_cuda::common::DeviceAccessible<Self::CudaRepresentation>,
        rust_cuda::host::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    )> {
        let alloc_front = rust_cuda::host::NullCudaAlloc;
        let alloc_tail = alloc;
        let (field_inner_repr, alloc_front) =
            rust_cuda::common::RustToCuda::borrow(&self.inner, alloc_front)?;
        let borrow = OuterCudaRepresentation {
            inner: field_inner_repr,
        };
        Ok((
            rust_cuda::common::DeviceAccessible::from(borrow),
            rust_cuda::host::CombinedCudaAlloc::new(alloc_front, alloc_tail),
        ))
    }

    #[cfg(not(target_os = "cuda"))]
    unsafe fn restore<CudaAllocType: rust_cuda::host::CudaAlloc>(
        &mut self,
        alloc: rust_cuda::host::CombinedCudaAlloc<Self::CudaAllocation, CudaAllocType>,
    ) -> rust_cuda::rustacuda::error::CudaResult<CudaAllocType> {
        let (alloc_front, alloc_tail) = alloc.split();
        let alloc_front = rust_cuda::common::RustToCuda::restore(&mut self.inner, alloc_front)?;
        Ok(alloc_tail)
    }
}
unsafe impl<T> rust_cuda::common::CudaAsRust for OuterCudaRepresentation<T>
where
    T: Copy + rust_cuda::common::RustToCuda,
{
    type RustRepresentation = Outer<T>;
}
