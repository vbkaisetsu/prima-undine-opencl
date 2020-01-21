use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(BatchSumFwImpl, batch_sum_fw_kernel);
impl FunctionFwImpl for BatchSumFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let size = y.shape().size();
        let batch = x.shape().batch();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&batch)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &[g1 * self.wgs[0], 1, 1],
                Some([self.wgs[0], 1, 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn batch_sum_fw_test() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., -2., -4., -6., -8., -10., -12., -14., -16.,
        ];
        let y_data = vec![-1., -2., -3., -4., -5., -6., -7., -8.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2, 2]);
        y.alloc();
        dev.call_fw_impl("batch_sum_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }
}
