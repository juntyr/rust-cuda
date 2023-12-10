use std::{ffi::CStr, mem::ManuallyDrop};

use rustacuda::{error::CudaResult, function::Function, module::Module};

#[doc(cfg(feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub struct CudaKernel {
    module: ManuallyDrop<Box<Module>>,
    function: ManuallyDrop<Function<'static>>,
}

impl CudaKernel {
    /// # Errors
    ///
    /// Returns a `CudaError` if `ptx` is not a valid PTX source, or it does
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

impl Drop for CudaKernel {
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
