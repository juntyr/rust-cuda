use std::{ffi::CString, ops::Deref};

use super::{PtxElement, PtxJITCompiler, PtxJITResult, PtxLoadWidth};

const R_ASCII_BYTE: u8 = 0x72;
const S_ASCII_BYTE: u8 = 0x73;
const D_ASCII_BYTE: u8 = 0x64;

impl PtxJITCompiler {
    pub fn with_arguments(&mut self, arguments: Option<&[Option<&[u8]>]>) -> PtxJITResult {
        // Check if the arguments, cast as byte slices, are the same as the last cached
        //  ones
        #[allow(clippy::explicit_deref_methods)]
        let needs_recomputation = match (arguments, &self.last_arguments) {
            (None, None) => false,
            (Some(arguments), Some(last_arguments)) if arguments.len() == last_arguments.len() => {
                arguments
                    .iter()
                    .zip(last_arguments.iter())
                    .all(|(a, b)| match (a, b) {
                        (None, None) => false,
                        (Some(a), Some(b)) => *a != b.deref(),
                        _ => true,
                    })
            },
            _ => true,
        };

        // Recompute the PTX string, optionally with constant loads, with the new
        //  arguments
        if needs_recomputation {
            // Cache the new arguments
            self.last_arguments = arguments.map(|arguments| {
                arguments
                    .iter()
                    .map(|arg| arg.map(|bytes| bytes.to_owned().into_boxed_slice()))
                    .collect::<Vec<Option<Box<[u8]>>>>()
                    .into_boxed_slice()
            });

            let mut output_ptx = Vec::new();

            if let Some(args) = &self.last_arguments {
                // Some constant loads are required, rebuild PTX string from source and newly
                //  generated constant load instructions
                for element in self.ptx_slices.iter() {
                    match element {
                        PtxElement::CopiedSource { ptx } => output_ptx.extend_from_slice(ptx),
                        PtxElement::ConstLoad {
                            ptx,
                            parameter_index,
                            byte_offset,
                            load_width,
                            register,
                        } => {
                            // Only generate a constant load instructions if the arguments contain
                            //  the byte range requested by the load instruction
                            if let Some(Some(arg)) = args.get(*parameter_index) {
                                if let Some(bytes) = arg.get(
                                    *byte_offset
                                        ..byte_offset
                                            + match load_width {
                                                PtxLoadWidth::B2 => 2,
                                                PtxLoadWidth::B4 => 4,
                                                PtxLoadWidth::B8 => 8,
                                            },
                                ) {
                                    // Generate the mov instruction with the correct data type
                                    output_ptx.extend_from_slice("mov.".as_bytes());
                                    output_ptx.extend_from_slice(
                                        if register.contains(&R_ASCII_BYTE) {
                                            "u".as_bytes()
                                        } else {
                                            "f".as_bytes()
                                        },
                                    );
                                    output_ptx.extend_from_slice(
                                        if register.contains(&S_ASCII_BYTE) {
                                            "16".as_bytes()
                                        } else if register.contains(&D_ASCII_BYTE) {
                                            "64".as_bytes()
                                        } else {
                                            "32".as_bytes()
                                        },
                                    );

                                    output_ptx.extend_from_slice(" \t".as_bytes());

                                    // Append the destination register from the load instruction
                                    output_ptx.extend_from_slice(register);

                                    // Generate the hexadecimal constant in little-endian order
                                    output_ptx.extend_from_slice(", 0x".as_bytes());
                                    for byte in bytes.iter().rev() {
                                        output_ptx
                                            .extend_from_slice(format!("{:02X}", byte).as_bytes());
                                    }

                                    output_ptx.extend_from_slice(";".as_bytes());

                                    // Early continue to the next `PtxElement` to avoid else branch
                                    continue;
                                }
                            }

                            // else: const load generation failed, fall back to original PTX source
                            output_ptx.extend_from_slice(ptx);
                        },
                    }
                }
            } else {
                // No constant loads are requires, just rebuild the PTX string from its slices
                for element in self.ptx_slices.iter() {
                    match element {
                        PtxElement::CopiedSource { ptx } | PtxElement::ConstLoad { ptx, .. } => {
                            output_ptx.extend_from_slice(ptx);
                        },
                    }
                }
            }

            // Cache the newly built PTX string
            self.last_ptx = unsafe { CString::from_vec_unchecked(output_ptx) };
        }

        if needs_recomputation {
            PtxJITResult::Recomputed(&self.last_ptx)
        } else {
            PtxJITResult::Cached(&self.last_ptx)
        }
    }
}
