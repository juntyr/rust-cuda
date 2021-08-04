use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, KernelConfig};

pub(super) fn quote_new_kernel(
    KernelConfig { kernel, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FuncIdent { func_ident, .. }: &FuncIdent,
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    quote! {
        fn new_kernel() -> rust_cuda::rustacuda::error::CudaResult<
            rust_cuda::host::TypedKernel<dyn #kernel #generic_start_token
                #($#macro_type_ids),*
            #generic_close_token>
        > {
            #[repr(C)]
            struct TypedKernel {
                compiler: rust_cuda::ptx_jit::host::compiler::PtxJITCompiler,
                kernel: Option<rust_cuda::ptx_jit::host::kernel::CudaKernel>,
                entry_point: Box<[u8]>,
            }

            let ptx_cstring = ::std::ffi::CString::new(Self::get_ptx_str())
                .map_err(|_| rust_cuda::rustacuda::error::CudaError::InvalidPtx)?;

            let compiler = rust_cuda::ptx_jit::host::compiler::PtxJITCompiler::new(
                &ptx_cstring
            );

            let entry_point_str = rust_cuda::host::specialise_kernel_call!(
                #func_ident #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token
            );
            let entry_point_cstring = ::std::ffi::CString::new(entry_point_str)
                .map_err(|_| rust_cuda::rustacuda::error::CudaError::UnknownError)?;
            let entry_point = entry_point_cstring
                .into_bytes_with_nul()
                .into_boxed_slice();

            let typed_kernel = TypedKernel { compiler, kernel: None, entry_point };

            Ok(unsafe { ::std::mem::transmute(typed_kernel) })
        }
    }
}
