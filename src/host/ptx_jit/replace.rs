use std::{ffi::CString, ops::Deref, ptr::NonNull};

use super::{PtxElement, PtxJITCompiler, PtxJITResult, PtxLoadWidth};

impl PtxJITCompiler {
    #[allow(clippy::too_many_lines)]
    pub fn with_arguments(&mut self, arguments: Option<&[Option<&NonNull<[u8]>>]>) -> PtxJITResult {
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
                        (Some(a), Some(b)) => (unsafe { a.as_ref() }) != b.deref(),
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
                    .map(|arg| {
                        arg.map(|bytes| unsafe { bytes.as_ref() }.to_owned().into_boxed_slice())
                    })
                    .collect::<Vec<Option<Box<[u8]>>>>()
                    .into_boxed_slice()
            });

            let mut output_ptx = Vec::new();
            let mut buffer_ptx = Vec::new();

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
                            registers,
                        } => {
                            let mut byte_offset = *byte_offset;
                            let mut registers_iter = registers.iter();

                            let Some(mut register) = registers_iter.next() else {
                                continue;
                            };

                            loop {
                                // Only generate a constant load instructions if the arguments
                                // contain  the byte range requested
                                // by the load instruction
                                if let Some(Some(arg)) = args.get(*parameter_index) {
                                    let load_width_bytes = match load_width {
                                        PtxLoadWidth::B1 => 1,
                                        PtxLoadWidth::B2 => 2,
                                        PtxLoadWidth::B4 => 4,
                                        PtxLoadWidth::B8 => 8,
                                    };

                                    if let Some(bytes) =
                                        arg.get(byte_offset..(byte_offset + load_width_bytes))
                                    {
                                        byte_offset += load_width_bytes;

                                        // Generate the mov instruction with the correct data type
                                        buffer_ptx.extend_from_slice(b"mov.");
                                        buffer_ptx.push(if register.contains(&b'r') {
                                            b'u'
                                        } else {
                                            b'f'
                                        });
                                        buffer_ptx.extend_from_slice(if register.contains(&b's') {
                                            b"16"
                                        } else if register.contains(&b'd') {
                                            b"64"
                                        } else {
                                            b"32"
                                        });

                                        buffer_ptx.extend_from_slice(b" \t");

                                        // Append the destination register from the load
                                        //  instruction
                                        buffer_ptx.extend_from_slice(register);

                                        // Generate the hexadecimal constant in little-endian
                                        //  order
                                        buffer_ptx.extend_from_slice(b", 0");
                                        buffer_ptx.push(if register.contains(&b'r') {
                                            b'x'
                                        } else if register.contains(&b'd') {
                                            b'd'
                                        } else {
                                            b'f'
                                        });
                                        for byte in bytes.iter().rev() {
                                            buffer_ptx
                                                .push(b"0123456789ABCDEF"[usize::from(*byte >> 4)]);
                                            buffer_ptx.push(
                                                b"0123456789ABCDEF"[usize::from(*byte & 0x0F_u8)],
                                            );
                                        }
                                        if register.contains(&b'r') {
                                            buffer_ptx.push(b'U');
                                        }

                                        buffer_ptx.push(b';');

                                        if let Some(next_register) = registers_iter.next() {
                                            // Early continue to the next `PtxElement` to avoid
                                            //  else branch
                                            register = next_register;
                                            buffer_ptx.push(b' ');
                                            continue;
                                        }

                                        // const load generation finished successfully
                                        // flush the generated instruction(s)
                                        output_ptx.append(&mut buffer_ptx);
                                        break;
                                    }
                                }

                                // else: const load generation failed
                                //       fall back to original PTX source
                                output_ptx.extend_from_slice(ptx);
                                buffer_ptx.clear();
                                break;
                            }
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
