use std::ffi::c_void;

#[link(name = "clblast", kind = "dylib")]
extern "C" {
    pub fn CLBlastSgemm(
        layout: i32,
        a_transpose: i32,
        b_transpose: i32,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_buffer: *const c_void,
        a_offset: usize,
        a_ld: usize,
        b_buffer: *const c_void,
        b_offset: usize,
        b_ld: usize,
        beta: f32,
        c_buffer: *mut c_void,
        c_offset: usize,
        c_ld: usize,
        queue: *mut *mut c_void,
        event: *mut *mut c_void,
    ) -> i32;

    pub fn CLBlastSgemmBatched(
        layout: i32,
        a_transpose: i32,
        b_transpose: i32,
        m: usize,
        n: usize,
        k: usize,
        alphas: *const f32,
        a_buffer: *const c_void,
        a_offsets: *const usize,
        a_ld: usize,
        b_buffer: *const c_void,
        b_offsets: *const usize,
        b_ld: usize,
        betas: *const f32,
        c_buffer: *mut c_void,
        c_offsets: *const usize,
        c_ld: usize,
        batch_count: usize,
        queue: *mut *mut c_void,
        event: *mut *mut c_void,
    ) -> i32;
}

#[allow(dead_code)]
pub mod layout {
    pub const ROW_MAJOR: i32 = 101;
    pub const COL_MAJOR: i32 = 102;
}

#[allow(dead_code)]
pub mod transpose {
    pub const NO: i32 = 111;
    pub const YES: i32 = 112;
    pub const CONJUGATE: i32 = 113;
}
