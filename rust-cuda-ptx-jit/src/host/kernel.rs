use std::{ffi::CStr, mem::ManuallyDrop};

use rustacuda::{error::CudaResult, function::Function, memory::DeviceCopy, module::{Module, Symbol}};

#[allow(clippy::module_name_repetitions)]
pub struct CudaKernel {
    module: ManuallyDrop<Box<Module>>,
    function: ManuallyDrop<Function<'static>>,
}

impl CudaKernel {
    /// # Errors
    ///
    /// Returns a `CudaError` if `ptx` is not a valid PTX source, or it does
    ///  not contain an entry point named `kernel`.
    pub fn new(ptx: &CStr, kernel: &CStr) -> CudaResult<Self> {
        let module = Box::leak(Box::new(Module::load_from_string(ptx)?)) as &Module;

        let function = module.get_function(kernel);

        let module = module as *const Module as *mut Module;
        let module = unsafe { Box::from_raw(module) };

        let function = match function {
            Ok(function) => function,
            Err(err) => {
                if let Err((_err, module)) = Module::drop(*module) {
                    std::mem::forget(module);
                }

                return Err(err);
            }
        };

        Ok(Self {
            function: ManuallyDrop::new(function),
            module: ManuallyDrop::new(module),
        })
    }

    /// # Errors
    ///
    /// Returns a `CudaError` if no symbol with `name` exists.
    pub fn get_module_global<T: DeviceCopy>(&self, name: &CStr) -> CudaResult<Symbol<T>> {
        self.module.get_global(name)
    }

    #[must_use]
    pub fn get_function(&self) -> &Function {
        &self.function
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        std::mem::drop(unsafe { ManuallyDrop::take(&mut self.function) });

        if let Err((_err, module)) = Module::drop(*unsafe { ManuallyDrop::take(&mut self.module) }) {
            std::mem::forget(module);
        }
    }
}
