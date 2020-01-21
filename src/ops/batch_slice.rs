use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(BatchSliceFwImpl, batch_slice_fw_kernel);
impl FunctionFwImpl for BatchSliceFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let offset = u32data[0];
        let y = &mut ys[0];
        let volume = y.shape().volume();
        let shift = volume * offset;
        let size = y.shape().size();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&shift)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size)).unwrap();
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

define_opencl_impl_struct!(BatchSliceBwImpl, batch_slice_bw_kernel);
impl FunctionBwImpl for BatchSliceBwImpl {
    fn call(
        &self,
        _xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let gy = gys[0];
        let offset = u32data[0];
        let volume = gy.shape().volume();
        let shift = volume * offset;
        let size = gy.shape().size();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(gx))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&shift)).unwrap();
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
    use prima_undine::Shape;

    #[test]
    fn check_batch_slice_fw() {
        let x_data = (0..18).map(|x| x as f32).collect::<Vec<f32>>();
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, 1, shape![3, 2], vec![0., 1., 2., 3., 4., 5.]),
            TestCase(1, 2, shape![3, 2], vec![6., 7., 8., 9., 10., 11.]),
            TestCase(2, 3, shape![3, 2], vec![12., 13., 14., 15., 16., 17.]),
            TestCase(
                0,
                2,
                shape![3, 2; 2],
                vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            ),
            TestCase(
                1,
                3,
                shape![3, 2; 2],
                vec![6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.],
            ),
            TestCase(0, 3, shape![3, 2; 3], x_data.clone()),
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 2; 3], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "batch_slice_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, &y.to_vec());
        }
    }

    #[test]
    fn check_batch_slice_bw_nn_1() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let gy_data = vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.];
        let gx_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let dev = get_device();
        let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 3], &gy_data);
        dev.call_bw_impl("batch_slice_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, &gx.to_vec());
    }

    #[test]
    fn check_batch_slice_bw_nn_2() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let gy_data = vec![1., 1., 2., 2., 3., 3., 4., 4.];
        struct TestCase(Shape, u32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2; 2],
                0,
                vec![1., 2., 4., 5., 3., 4., 6., 7., 0., 1., 2., 3.],
            ),
            TestCase(
                shape![2, 2; 2],
                1,
                vec![0., 1., 2., 3., 1., 2., 4., 5., 3., 4., 6., 7.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl(
                "batch_slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.2, &gx.to_vec());
        }
    }

    #[test]
    fn check_batch_slice_bw_n1_1() {
        let a_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let gy_data = vec![-1., -2., -3., -4.];
        let gx_data = vec![0., 0., 0., 0., 2., 3., 4., 5., 3., 4., 5., 6.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
        dev.call_bw_impl("batch_slice_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, &gx.to_vec());
    }

    #[test]
    fn check_batch_slice_bw_n1_2() {
        let a_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let gy_data = vec![-1., -2., -3., -4.];
        struct TestCase(Shape, u32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2],
                0,
                vec![0., 0., 0., 0., 2., 3., 4., 5., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 2],
                1,
                vec![1., 2., 3., 4., 1., 1., 1., 1., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 2],
                2,
                vec![1., 2., 3., 4., 2., 3., 4., 5., 2., 2., 2., 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl(
                "batch_slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.2, &gx.to_vec());
        }
    }
}
