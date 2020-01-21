use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(MulAssignConstImpl, mul_assign_const_kernel);
impl FunctionFwImpl for MulAssignConstImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        let size = y.shape().size();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&k)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(y))).unwrap();
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
    fn check_mul_assign_const() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 5.;
        let y_data = vec![5000., 500., 50., 5., 0.5, 0.05, 0.005, 0.0005];
        let dev = get_device();
        let mut x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        dev.call_fw_impl("mul_assign_const_impl", &[], &[], &[k], &mut [&mut x]);
        assert_vector_ulps_eq!(y_data, x.to_vec());
    }
}
