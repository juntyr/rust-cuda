use std::time::Duration;

use fork::{fork, Fork};
use sysinfo::{RefreshKind, System, SystemExt};

fn main() {
    const DREAM: Duration = Duration::from_millis(1000);

    let root_pid = std::process::id();

    println!("ROOT PID: {}", root_pid);

    match fork().unwrap() {
        Fork::Parent(_) => loop {
            std::thread::sleep(DREAM);
        },
        Fork::Child => {
            unsafe {
                libc::setsid();
            }

            let mut command = std::process::Command::new(std::env::args_os().nth(1).unwrap());
            command.args(std::env::args_os().skip(2));

            let mut child = command.spawn().unwrap();

            let mut system = System::new_with_specifics(RefreshKind::new());

            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }

                if !system.refresh_process(root_pid as i32) {
                    unsafe {
                        // TODO: Deal with zombie processes
                        libc::kill(-(std::process::id() as i32), libc::SIGINT);
                    }
                } else {
                    std::thread::sleep(DREAM);
                }
            };

            unsafe {
                libc::kill(root_pid as i32, libc::SIGINT);
            }

            if let Some(code) = status.code() {
                std::process::exit(code);
            } else {
                std::process::abort();
            }
        },
    }
}
