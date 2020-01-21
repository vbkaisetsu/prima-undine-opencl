use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(BroadcastFwImpl, broadcast_fw_kernel);
impl FunctionFwImpl for BroadcastFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let size = u32data[1];
        let y = &mut ys[0];
        let skip1 = y.shape().lower_volume(dim);
        let skip2 = skip1 * size;
        let total = y.shape().size();
        let g1 = super::common::calc_num_blocks(total as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&skip1)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&skip2)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&total)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(y))).unwrap();
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
    fn check_broadcast_fw() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, 1, shape![], vec![1.]),
            TestCase(0, 20, shape![20], vec![1.; 20]),
            TestCase(1, 50, shape![1, 50], vec![1.; 50]),
            TestCase(2, 100, shape![1, 1, 100], vec![1.; 100]),
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_constant(shape![], 1.);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }

    #[test]
    fn check_broadcast_fw_2() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(1, 1, shape![2; 3], vec![1., 2., 3., 4., 5., 6.]),
            TestCase(2, 1, shape![2; 3], vec![1., 2., 3., 4., 5., 6.]),
            TestCase(
                1,
                2,
                shape![2, 2; 3],
                vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6.],
            ),
            TestCase(
                2,
                2,
                shape![2, 1, 2; 3],
                vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6.],
            ),
        ];
        let x_data = vec![1., 2., 3., 4., 5., 6.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2; 3], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }

    #[test]
    fn check_broadcast_fw_3() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                2,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                4,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                0,
                2,
                shape![2, 2, 1, 2; 2],
                vec![
                    1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7., 8., 8.,
                ],
            ),
            TestCase(
                2,
                2,
                shape![1, 2, 2, 2; 2],
                vec![
                    1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7., 8.,
                ],
            ),
            TestCase(
                4,
                2,
                shape![1, 2, 1, 2, 2; 2],
                vec![
                    1., 2., 3., 4., 1., 2., 3., 4., 5., 6., 7., 8., 5., 6., 7., 8.,
                ],
            ),
        ];
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![1, 2, 1, 2; 2], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }
}
