use thiserror::Error;

#[allow(non_camel_case_types)]
pub type size_t = ::std::os::raw::c_ulonglong;

#[repr(C)]
pub struct nvPTXCompiler {
    _private: [u8; 0],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum NvptxError {
    #[error("Invalid compiler handle")]
    InvalidCompilerHandle,
    #[error("Invalid PTX input")]
    InvalidInput,
    #[error("Compilation failure")]
    CompilationFailure,
    #[error("Internal error")]
    Internal,
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Incomplete compiler invocation")]
    CompilerInvocationIncomplete,
    #[error("Unsupported PTX version")]
    UnsupportedPtxVersion,
    #[error("Unsupported dev-side sync")]
    UnsupportedDevSideSync,
    #[error("Unknown error code")]
    UnknownError,
}

impl NvptxError {
    const NVPTXCOMPILE_ERROR_COMPILATION_FAILURE: NvptxCompileResult = 3;
    const NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE: NvptxCompileResult = 6;
    const NVPTXCOMPILE_ERROR_INTERNAL: NvptxCompileResult = 4;
    const NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE: NvptxCompileResult = 1;
    const NVPTXCOMPILE_ERROR_INVALID_INPUT: NvptxCompileResult = 2;
    const NVPTXCOMPILE_ERROR_OUT_OF_MEMORY: NvptxCompileResult = 5;
    const NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC: NvptxCompileResult = 8;
    const NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION: NvptxCompileResult = 7;
    const NVPTXCOMPILE_SUCCESS: NvptxCompileResult = 0;

    pub fn try_err_from(result: NvptxCompileResult) -> Result<(), Self> {
        match result {
            Self::NVPTXCOMPILE_SUCCESS => Ok(()),
            Self::NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE => Err(Self::InvalidCompilerHandle),
            Self::NVPTXCOMPILE_ERROR_INVALID_INPUT => Err(Self::InvalidInput),
            Self::NVPTXCOMPILE_ERROR_COMPILATION_FAILURE => Err(Self::CompilationFailure),
            Self::NVPTXCOMPILE_ERROR_INTERNAL => Err(Self::Internal),
            Self::NVPTXCOMPILE_ERROR_OUT_OF_MEMORY => Err(Self::OutOfMemory),
            Self::NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE => {
                Err(Self::CompilerInvocationIncomplete)
            },
            Self::NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION => Err(Self::UnsupportedPtxVersion),
            Self::NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC => Err(Self::UnsupportedDevSideSync),
            _ => Err(Self::UnknownError),
        }
    }
}

/// [`nvPTXCompilerHandle`] represents a handle to the PTX Compiler.
///
/// To compile a PTX program string, an instance of [`nvPTXCompiler`]
/// must be created and the handle to it must be obtained using the
/// API [`nvPTXCompilerCreate`]. Then the compilation can be done
/// using the API [`nvPTXCompilerCompile`].
pub type NvptxCompilerHandle = *mut nvPTXCompiler;

/// The [`nvPTXCompiler`] APIs return the [`nvPTXCompileResult`] codes to
/// indicate the call result"]
pub type NvptxCompileResult = ::std::os::raw::c_int;

extern "C" {
    /// Queries the current major and minor version of PTX Compiler APIs being
    /// used.
    ///
    /// # Parameters
    /// - [out] `major`: Major version of the PTX Compiler APIs
    /// - [out] `minor`: Minor version of the PTX Compiler APIs
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    ///
    /// # Note
    /// The version of PTX Compiler APIs follows the CUDA Toolkit versioning.
    /// The PTX ISA version supported by a PTX Compiler API version is listed
    /// [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes).
    pub fn nvPTXCompilerGetVersion(
        major: *mut ::std::os::raw::c_uint,
        minor: *mut ::std::os::raw::c_uint,
    ) -> NvptxCompileResult;

    /// Obtains the handle to an instance of the PTX compiler
    /// initialized with the given PTX program `ptxCode`.
    ///
    /// # Parameters
    /// - [out] `compiler`: Returns a handle to PTX compiler initialized with
    ///   the PTX program `ptxCode`
    /// - [in] `ptxCodeLen`: Size of the PTX program `ptxCode` passed as a
    ///   string
    /// - [in] `ptxCode`: The PTX program which is to be compiled passed as a
    ///   string
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_OUT_OF_MEMORY`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    pub fn nvPTXCompilerCreate(
        compiler: *mut NvptxCompilerHandle,
        ptxCodeLen: size_t,
        ptxCode: *const ::std::os::raw::c_char,
    ) -> NvptxCompileResult;

    /// Destroys and cleans the already created PTX compiler.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to the PTX compiler which is to be
    ///   destroyed.
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_OUT_OF_MEMORY`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    pub fn nvPTXCompilerDestroy(compiler: *mut NvptxCompilerHandle) -> NvptxCompileResult;

    /// Compile a PTX program with the given compiler options.
    ///
    /// # Parameters
    /// - [in, out] `compiler`: A handle to PTX compiler initialized with the
    ///   PTX program which is to be compiled. The compiled program can be
    ///   accessed using the handle.
    /// - [in] `numCompileOptions`: Length of the array `compileOptions`
    /// - [in] `compileOptions`: Compiler options with which compilation should
    ///   be done. The compiler options string is a null terminated character
    ///   array. A valid list of compiler options is available at
    ///   [link](http://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compile-options).
    ///
    /// # Note
    /// `--gpu-name` (`-arch`) is a mandatory option.
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_OUT_OF_MEMORY`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_COMPILATION_FAILURE`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION`]
    pub fn nvPTXCompilerCompile(
        compiler: NvptxCompilerHandle,
        numCompileOptions: ::std::os::raw::c_int,
        compileOptions: *const *const ::std::os::raw::c_char,
    ) -> NvptxCompileResult;

    /// Obtains the size of the image of the compiled program.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `binaryImageSize`: The size of the image of the compiled program
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`]
    ///
    /// # Note
    /// The [`nvPTXCompilerCompile`] function should be invoked for the handle
    /// before calling this API. Otherwise,
    /// [`NvptxCompileResult::NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`]
    /// is returned.
    pub fn nvPTXCompilerGetCompiledProgramSize(
        compiler: NvptxCompilerHandle,
        binaryImageSize: *mut size_t,
    ) -> NvptxCompileResult;

    /// Obtains the image of the compiled program.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `binaryImage`: The image of the compiled program. The caller
    ///   should allocate memory for `binaryImage`.
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`]
    ///
    /// # Note
    /// The [`nvPTXCompilerCompile`] function should be invoked for the handle
    /// before calling this API. Otherwise,
    /// [`NvptxCompileResult::NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`]
    /// is returned.
    pub fn nvPTXCompilerGetCompiledProgram(
        compiler: NvptxCompilerHandle,
        binaryImage: *mut ::std::os::raw::c_void,
    ) -> NvptxCompileResult;

    /// Query the size of the error message that was seen previously for the
    /// handle.
    ///
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `errorLogSize`: The size of the error log in bytes which was
    ///   produced in previous call to [`nvPTXCompilerCompile`].
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    pub fn nvPTXCompilerGetErrorLogSize(
        compiler: NvptxCompilerHandle,
        errorLogSize: *mut size_t,
    ) -> NvptxCompileResult;

    /// Query the error message that was seen previously for the handle.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `errorLog`: The error log which was produced in previous call to
    ///   [`nvPTXCompilerCompile`]. The caller should allocate memory for
    ///   `errorLog`.
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    pub fn nvPTXCompilerGetErrorLog(
        compiler: NvptxCompilerHandle,
        errorLog: *mut ::std::os::raw::c_char,
    ) -> NvptxCompileResult;

    /// Query the size of the information message that was seen previously for
    /// the handle.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `infoLogSize`: The size of the information log in bytes which
    ///   was produced in previous call to [`nvPTXCompilerCompile`].
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    pub fn nvPTXCompilerGetInfoLogSize(
        compiler: NvptxCompilerHandle,
        infoLogSize: *mut size_t,
    ) -> NvptxCompileResult;

    /// Query the information message that was seen previously for the handle.
    ///
    /// # Parameters
    /// - [in] `compiler`: A handle to PTX compiler on which
    ///   [`nvPTXCompilerCompile`] has been performed.
    /// - [out] `infoLog`: The information log which was produced in previous
    ///   call to [`nvPTXCompilerCompile`]. The caller should allocate memory
    ///   for `infoLog`.
    ///
    /// # Returns
    /// - [`NvptxCompileResult::NVPTXCOMPILE_SUCCESS`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INTERNAL`]
    /// - [`NvptxCompileResult::NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE`]
    pub fn nvPTXCompilerGetInfoLog(
        compiler: NvptxCompilerHandle,
        infoLog: *mut ::std::os::raw::c_char,
    ) -> NvptxCompileResult;
}
