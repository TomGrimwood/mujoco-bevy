//! Access `MuJoCo`'s exported data on Windows.
//!
//! The import library only has `__imp_<name>` for DLL data. mujoco-rs's
//! `extern static mut` is not marked `dllimport`, so a bare `mjTIMERSTRING`
//! (and the other tables / `mjcb_*` slots) fails to link. Reading through
//! the import thunk reaches the real symbol in `mujoco.dll`.

#[cfg(test)]
use std::ffi::c_char;

use mujoco_rs::mujoco_c;

/// The process-wide `mjcb_time` slot.
///
/// # Safety
/// Same as reading/writing `mujoco_c::mjcb_time`: only while nothing steps.
#[must_use]
pub unsafe fn mjcb_time() -> *mut mujoco_c::mjfTime {
    #[cfg(windows)]
    {
        unsafe extern "C" {
            #[link_name = "__imp_mjcb_time"]
            static mut IMP: *mut mujoco_c::mjfTime;
        }
        unsafe { IMP }
    }
    #[cfg(not(windows))]
    {
        &raw mut mujoco_c::mjcb_time
    }
}

/// A reader of one of `MuJoCo`'s constant string tables (the tests check
/// names against them).
#[cfg(test)]
macro_rules! dll_table {
    ($fn:ident, $sym:ident, $ty:ty) => {
        /// # Safety
        /// Read of a `MuJoCo` constant string table for the life of the process.
        #[must_use]
        pub unsafe fn $fn() -> &'static $ty {
            #[cfg(windows)]
            {
                unsafe extern "C" {
                    #[link_name = concat!("__imp_", stringify!($sym))]
                    static mut IMP: *mut $ty;
                }
                unsafe { &*IMP }
            }
            #[cfg(not(windows))]
            {
                // Through a raw pointer: a reference to a `static mut`
                // straight is refused (Rust 2024).
                unsafe { &*(&raw const mujoco_c::$sym) }
            }
        }
    };
}

#[cfg(test)]
dll_table!(mj_timer_string, mjTIMERSTRING, [*const c_char; 15]);
#[cfg(test)]
dll_table!(mj_disable_string, mjDISABLESTRING, [*const c_char; 20]);
#[cfg(test)]
dll_table!(mj_enable_string, mjENABLESTRING, [*const c_char; 7]);
#[cfg(test)]
dll_table!(mj_label_string, mjLABELSTRING, [*const c_char; 17]);
#[cfg(test)]
dll_table!(mj_frame_string, mjFRAMESTRING, [*const c_char; 8]);
#[cfg(test)]
dll_table!(mj_vis_string, mjVISSTRING, [[*const c_char; 3]; 31]);
#[cfg(test)]
dll_table!(mj_rnd_string, mjRNDSTRING, [[*const c_char; 3]; 11]);
