use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(BatchConcatFwImpl, batch_concat_fw_kernel);
impl FunctionFwImpl for BatchConcatFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let y = &mut ys[0];
        let mut offset = 0;
        for x in xs {
            let span = x.shape().size();
            let g1 = super::common::calc_num_blocks(span as usize, self.wgs[0]);
            let queue = &self.internal.queue;
            let kernel = self.kernel.lock().unwrap();
            unsafe {
                ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&span)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(y))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&offset)).unwrap();
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
            offset += span;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_batch_concat_fw() {
        let a_data = vec![1., 2., 3., 4., 5., 6.];
        let b_data = vec![7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.];
        let c_data = vec![
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.,
            36.,
        ];
        let y_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 3], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 3; 2], &b_data);
        let c = dev.new_tensor_by_slice(shape![2, 3; 3], &c_data);
        let mut y = dev.new_tensor(shape![2, 3; 6]);
        y.alloc();
        dev.call_fw_impl(
            "batch_concat_fw_impl",
            &[&a, &b, &c],
            &[],
            &[],
            &mut [&mut y],
        );
        assert_vector_ulps_eq!(&y_data, &y.to_vec());
    }
}
