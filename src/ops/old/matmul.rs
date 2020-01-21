use std::ptr;

use ocl::Queue;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

use crate::clblast;

pub struct MatmulFwImpl {
    queue: Queue,
}

impl MatmulFwImpl {
    pub fn new(queue: Queue) -> MatmulFwImpl {
        MatmulFwImpl { queue: queue }
    }
}

impl FunctionFwImpl for MatmulFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let a = xs[0];
        let b = xs[1];
        let y = &mut ys[0];
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![0.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            unsafe {
                clblast::CLBlastSgemmBatched(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::NO,
                    di,
                    dk,
                    dj,
                    alphas.as_ptr(),
                    buffer!(a).as_ptr(),
                    a_offsets.as_ptr(),
                    di,
                    buffer!(b).as_ptr(),
                    b_offsets.as_ptr(),
                    dj,
                    betas.as_ptr(),
                    buffer!(y).as_ptr(),
                    y_offsets.as_ptr(),
                    di,
                    bs,
                    &mut self.queue.as_ptr(),
                    &mut ptr::null_mut(),
                );
            }
        } else {
            let alpha = 1.;
            let beta = 0.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::NO,
                    di,
                    dk * b.shape().batch() as usize,
                    dj,
                    alpha,
                    buffer!(a).as_ptr(),
                    0,
                    di,
                    buffer!(b).as_ptr(),
                    0,
                    dj,
                    beta,
                    buffer!(y).as_ptr(),
                    0,
                    di,
                    &mut self.queue.as_ptr(),
                    &mut ptr::null_mut(),
                );
            }
        }
    }
}

pub struct MatmulBwAImpl {
    queue: Queue,
}

impl MatmulBwAImpl {
    pub fn new(queue: Queue) -> MatmulBwAImpl {
        MatmulBwAImpl { queue: queue }
    }
}

impl FunctionBwImpl for MatmulBwAImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let b = xs[1];
        let gy = gys[0];
        let ga = gx;
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![1.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            unsafe {
                clblast::CLBlastSgemmBatched(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::YES,
                    di,
                    dj,
                    dk,
                    alphas.as_ptr(),
                    buffer!(gy).as_ptr(),
                    y_offsets.as_ptr(),
                    dj,
                    buffer!(b).as_ptr(),
                    b_offsets.as_ptr(),
                    dj,
                    betas.as_ptr(),
                    buffer!(ga).as_ptr(),
                    a_offsets.as_ptr(),
                    di,
                    bs,
                    &mut self.queue.as_ptr(),
                    &mut ptr::null_mut(),
                );
            }
        } else {
            let alpha = 1.;
            let beta = 1.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::YES,
                    di,
                    dj,
                    dk * b.shape().batch() as usize,
                    alpha,
                    buffer!(gy).as_ptr(),
                    0,
                    di,
                    buffer!(b).as_ptr(),
                    0,
                    dj,
                    beta,
                    buffer!(ga).as_ptr(),
                    0,
                    di,
                    &mut self.queue.as_ptr(),
                    &mut ptr::null_mut(),
                );
            }
        }
    }
}

pub struct MatmulBwBImpl {
    queue: Queue,
}

impl MatmulBwBImpl {
    pub fn new(queue: Queue) -> MatmulBwBImpl {
        MatmulBwBImpl { queue: queue }
    }
}

impl FunctionBwImpl for MatmulBwBImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let b = xs[1];
        let gy = gys[0];
        let gb = gx;
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![1.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            if b_skip > 0 {
                unsafe {
                    clblast::CLBlastSgemmBatched(
                        clblast::layout::COL_MAJOR,
                        clblast::transpose::YES,
                        clblast::transpose::NO,
                        dj,
                        dk,
                        di,
                        alphas.as_ptr(),
                        buffer!(a).as_ptr(),
                        a_offsets.as_ptr(),
                        di,
                        buffer!(gy).as_ptr(),
                        y_offsets.as_ptr(),
                        di,
                        betas.as_ptr(),
                        buffer!(gb).as_ptr(),
                        b_offsets.as_ptr(),
                        dj,
                        bs,
                        &mut self.queue.as_ptr(),
                        &mut ptr::null_mut(),
                    );
                }
            } else {
                let alpha = 1.;
                let beta = 1.;
                for n in 0..bs {
                    unsafe {
                        clblast::CLBlastSgemm(
                            clblast::layout::COL_MAJOR,
                            clblast::transpose::YES,
                            clblast::transpose::NO,
                            dj,
                            dk,
                            di,
                            alpha,
                            buffer!(a).as_ptr(),
                            n * a_skip,
                            di,
                            buffer!(gy).as_ptr(),
                            n * y_skip,
                            di,
                            beta,
                            buffer!(gb).as_ptr(),
                            n * b_skip,
                            dj,
                            &mut self.queue.as_ptr(),
                            &mut ptr::null_mut(),
                        );
                    }
                }
            }
        } else {
            let alpha = 1.;
            let beta = 1.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::YES,
                    clblast::transpose::NO,
                    dj,
                    dk * b.shape().batch() as usize,
                    di,
                    alpha,
                    buffer!(a).as_ptr(),
                    0,
                    di,
                    buffer!(gy).as_ptr(),
                    0,
                    di,
                    beta,
                    buffer!(gb).as_ptr(),
                    0,
                    dj,
                    &mut self.queue.as_ptr(),
                    &mut ptr::null_mut(),
                );
            }
        }
    }
}
