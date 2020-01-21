use prima_undine::device_impl::FunctionFwF32Impl;
use prima_undine::Tensor;

define_empty_impl!(TensorToVectorImpl);
impl FunctionFwF32Impl for TensorToVectorImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [f32]) {
        let x = &xs[0];
        unsafe {
            super::common::read_buffer(&self.internal.queue, buffer!(x), ys);
        }
    }
}
