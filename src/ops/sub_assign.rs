use std::cmp;

use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(SubAssignImpl, sub_assign_kernel);
impl FunctionFwImpl for SubAssignImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let size = y.shape().volume();
        let mbx = x.shape().has_batch() as u32;
        let mby = y.shape().has_batch() as u32;
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let g2 = cmp::max(x.shape().batch(), y.shape().batch()) as usize;
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&mbx)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&mby)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                2,
                None,
                &[g1 * self.wgs[0], g2, 1],
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
    fn check_sub_assign() {
        let a_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let b_data = vec![0., 100., 20., 3., 0.4, 0.05, 0.006, 0.0007];
        let y_data = vec![-1000., 0., 10., 2., 0.3, 0.04, 0.005, 0.0006];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let mut b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        dev.call_fw_impl("sub_assign_impl", &[&a], &[], &[], &mut [&mut b]);
        assert_vector_ulps_eq!(y_data, b.to_vec());
    }

    #[test]
    fn check_sub_assign_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![-2., -2., -2., -2., 4., 4., 4., 4.];
        let y1_data = vec![-2., -3., -4., -5., 4., 3., 2., 1.];
        let y2_data = vec![-2., -1., 0., 1.];
        let dev = get_device();
        {
            let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
            let mut b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            dev.call_fw_impl("sub_assign_impl", &[&a], &[], &[], &mut [&mut b]);
            assert_vector_ulps_eq!(y1_data, b.to_vec());
        }
        {
            let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            let mut a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
            dev.call_fw_impl("sub_assign_impl", &[&b], &[], &[], &mut [&mut a]);
            assert_vector_ulps_eq!(y2_data, a.to_vec());
        }
    }
}
