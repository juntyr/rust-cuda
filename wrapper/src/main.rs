#![deny(clippy::pedantic)]

use std::{cmp::Ordering, ffi::CString, os::unix::prelude::OsStrExt};

use anyhow::Context;
use fork::{fork, Fork};
use signal_hook::iterator::Signals;

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let mut args = std::env::args_os().skip(1).peekable();

    let program = match args.peek() {
        Some(program) => CString::new(program.as_bytes().to_owned())
            .with_context(|| format!("Program name {:?} is not a valid program name.", program))?,
        None => return Ok(()),
    };

    let args = args
        .map(|arg| {
            CString::new(arg.as_bytes().to_owned()).with_context(|| {
                format!(
                    "Program argument {:?} is not a valid program argument.",
                    arg
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let canary_pid = unsafe { libc::getpid() };

    match fork()
        .map_err(|_| std::io::Error::last_os_error())
        .context("Failed to fork the wrapper into the canary and handler.")?
    {
        Fork::Parent(handler_pid) => {
            let mut signals = Signals::new(
                (1..32).filter(|signal| !signal_hook::consts::FORBIDDEN.contains(signal)),
            )
            .context("Failed to install the signal handler in the canary.")?;

            let status = loop {
                let mut status = 0;

                match unsafe { libc::waitpid(handler_pid, &mut status, libc::WNOHANG) }.cmp(&0) {
                    Ordering::Less => {
                        return Err(std::io::Error::last_os_error())
                            .context("Failed to wait for the handler to terminate.")
                    },
                    Ordering::Equal => (),
                    Ordering::Greater => break libc::WEXITSTATUS(status),
                };

                for signal in signals.wait() {
                    log::debug!(
                        "The CANARY {} has received the signal {}.",
                        canary_pid,
                        signal
                    );

                    if signal != libc::SIGCHLD {
                        log::debug!(
                            "The CANARY {} will send the signal {} to the HANDLER {}.",
                            handler_pid,
                            signal,
                            handler_pid
                        );

                        // Error is ignored as the handler may have already terminated
                        unsafe { libc::kill(handler_pid, signal) };
                    }
                }
            };

            log::debug!(
                "The CANARY {} will exit with status {}.",
                canary_pid,
                status
            );

            std::process::exit(status);
        },
        Fork::Child => {
            let handler_pid = unsafe { libc::getpid() };

            unsafe {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGCHLD) == -1 {
                    return Err(std::io::Error::last_os_error())
                        .context("Failed to setup the death signal in the handler.");
                }
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error())
                        .context("Failed to create a new process group for the handler.");
                }
                if libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) == -1 {
                    return Err(std::io::Error::last_os_error())
                        .context("Failed to make the handler a sub-reaper for its descendants.");
                }
            }

            // Check if the canary quit before the handler, if so abort
            if unsafe { libc::getppid() } != canary_pid {
                log::warn!(
                    "The CANARY {} has been terminated before the HANDLER {} could start.",
                    canary_pid,
                    handler_pid
                );

                std::process::abort();
            }

            let mut signals = Signals::new(
                (1..32).filter(|signal| !signal_hook::consts::FORBIDDEN.contains(signal)),
            )
            .context("Failed to install the signal handler in the handler.")?;

            match fork()
                .map_err(|_| std::io::Error::last_os_error())
                .context("Failed to fork the wrapper into the handler and worker.")?
            {
                Fork::Parent(worker_pid) => {
                    let status = 'outer: loop {
                        let mut status = 0;

                        match unsafe { libc::waitpid(worker_pid, &mut status, libc::WNOHANG) }
                            .cmp(&0)
                        {
                            Ordering::Less => {
                                return Err(std::io::Error::last_os_error())
                                    .context("Failed to wait for the worker to terminate.")
                            },
                            Ordering::Equal => (),
                            Ordering::Greater => break status,
                        };

                        for signal in signals.wait() {
                            log::debug!(
                                "The HANDLER {} has received the signal {}.",
                                handler_pid,
                                signal
                            );

                            if signal != libc::SIGCHLD {
                                log::debug!(
                                    "The HANDLER {} will send the signal {} to the WORKER group \
                                     {}.",
                                    handler_pid,
                                    signal,
                                    worker_pid
                                );

                                unsafe {
                                    // Error is ignored as the worker may have already terminated
                                    libc::kill(-worker_pid, signal);
                                }
                            } else if unsafe { libc::getppid() } != canary_pid {
                                log::debug!(
                                    "The HANDLER {} has been informed that the CANARY {} has died.",
                                    handler_pid,
                                    canary_pid
                                );
                                log::debug!(
                                    "The HANDLER {} will terminate the WORKER group {}.",
                                    handler_pid,
                                    worker_pid
                                );

                                // Error is ignored as the worker may have already terminated
                                unsafe {
                                    libc::kill(-worker_pid, libc::SIGKILL);
                                }

                                log::debug!(
                                    "The HANDLER {} will wait for the WORKER leader {}.",
                                    handler_pid,
                                    worker_pid
                                );

                                let mut status = 0;
                                if unsafe { libc::waitpid(worker_pid, &mut status, 0) } > 0 {
                                    break 'outer status;
                                }

                                return Err(std::io::Error::last_os_error())
                                    .context("Failed to wait for the worker to terminate.");
                            }
                        }
                    };

                    log::debug!(
                        "The HANDLER {} will wait for the WORKER group {}.",
                        handler_pid,
                        worker_pid
                    );

                    // Error means that we haved waited for all children
                    while unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) } >= 0 {}

                    log::debug!(
                        "The HANDLER {} will exit with status {}.",
                        handler_pid,
                        status
                    );

                    std::process::exit(status);
                },
                Fork::Child => {
                    let worker_pid = unsafe { libc::getpid() };

                    if unsafe { libc::setsid() } == -1 {
                        return Err(std::io::Error::last_os_error())
                            .context("Failed to create a new process group for the worker.");
                    }

                    log::debug!(
                        "The WORKER {} will execute the {:?} command.",
                        worker_pid,
                        args
                    );

                    let mut args: Vec<*const libc::c_char> =
                        args.iter().map(|s| s.as_ptr()).collect();
                    args.push(std::ptr::null());

                    // `execvp` only returns on error
                    unsafe { libc::execvp(program.as_ptr(), args.as_ptr()) };

                    Err(std::io::Error::last_os_error())
                        .context("Failed to execute the command in the worker.")
                },
            }
        },
    }
}
