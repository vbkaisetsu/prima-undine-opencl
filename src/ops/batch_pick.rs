use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(BatchPickFwImpl, batch_pick_fw_kernel);
impl FunctionFwImpl for BatchPickFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let ids = u32data;
        let y = &mut ys[0];
        let si = (ids.len() > 1) as u32;
        let sy = y.shape().volume();
        let g1 = super::common::calc_num_blocks(sy as usize, self.wgs[0]);
        let bs = y.shape().batch() as usize;
        let ids_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                ids.len(),
                None::<&[u32]>,
            )
            .unwrap()
        };
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            super::common::write_buffer(&self.internal.queue, ids, &ids_buf);
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::mem(&ids_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&si)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&sy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                2,
                None,
                &[g1 * self.wgs[0], bs, 1],
                Some([self.wgs[0], 1, 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

define_opencl_impl_struct!(BatchPickBwImpl, batch_pick_bw_kernel);
impl FunctionBwImpl for BatchPickBwImpl {
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
        let ids = u32data;
        let si = (ids.len() > 1) as u32;
        let sy = gy.shape().volume();
        let g1 = super::common::calc_num_blocks(sy as usize, self.wgs[0]);
        let bs = gy.shape().batch() as usize;
        let ids_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                ids.len(),
                None::<&[u32]>,
            )
            .unwrap()
        };
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            super::common::write_buffer(&self.internal.queue, ids, &ids_buf);
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::mem(&ids_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&si)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&sy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(gx))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                2,
                None,
                &[g1 * self.wgs[0], bs, 1],
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
    fn check_batch_pick_fw() {
        struct TestCase(Shape, Vec<u32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2; 3],
                vec![0, 0, 0],
                shape![2, 2; 3],
                vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1, 0, 1],
                shape![2, 2; 3],
                vec![4., 5., 6., 7., 0., 1., 2., 3., 4., 5., 6., 7.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![2],
                shape![2, 2],
                vec![8., 9., 10., 11.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![2, 1],
                shape![2, 2; 2],
                vec![8., 9., 10., 11., 4., 5., 6., 7.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![2, 0, 1, 1],
                shape![2, 2; 4],
                vec![
                    8., 9., 10., 11., 0., 1., 2., 3., 4., 5., 6., 7., 4., 5., 6., 7.,
                ],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x_data = (0..tc.0.size()).map(|x| x as f32).collect::<Vec<f32>>();
            let x = dev.new_tensor_by_slice(tc.0, &x_data);
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl("batch_pick_fw_impl", &[&x], &tc.1, &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.3, &y.to_vec());
        }
    }

    #[test]
    fn check_batch_pick_bw_nn() {
        let a_data = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.];
        struct TestCase(Shape, Vec<f32>, Vec<u32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                vec![0, 0, 0],
                vec![6., 7., 8., 9., 4., 5., 6., 7., 8., 9., 10., 11.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                vec![1, 1, 2],
                vec![0., 1., 2., 3., 7., 8., 9., 10., 11., 12., 13., 14.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                vec![0, 1, 0],
                vec![4., 5., 6., 7., 6., 7., 8., 9., 8., 9., 10., 11.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                vec![1, 0, 1],
                vec![2., 3., 4., 5., 8., 9., 10., 11., 8., 9., 10., 11.],
            ),
            TestCase(
                shape![2, 2; 2],
                vec![1., 1., 1., 1., 2., 2., 2., 2.],
                vec![2, 1],
                vec![0., 1., 2., 3., 6., 7., 8., 9., 9., 10., 11., 12.],
            ),
            TestCase(
                shape![2, 2; 4],
                vec![
                    1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4.,
                ],
                vec![1, 2, 0, 1],
                vec![3., 4., 5., 6., 9., 10., 11., 12., 10., 11., 12., 13.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl("batch_pick_bw_impl", &[], &[], &[&gy], &tc.2, &[], &mut gx);
            assert_vector_ulps_eq!(tc.3, &gx.to_vec());
        }
    }

    #[test]
    fn check_batch_pick_bw_n1() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        struct TestCase(Shape, Vec<f32>, Vec<u32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2],
                vec![1., 1., 2., 2.],
                vec![0],
                vec![1., 2., 4., 5., 0., 1., 2., 3., 0., 1., 2., 3.],
            ),
            TestCase(
                shape![2, 2],
                vec![1., 1., 2., 2.],
                vec![1],
                vec![0., 1., 2., 3., 1., 2., 4., 5., 0., 1., 2., 3.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl("batch_pick_bw_impl", &[], &[], &[&gy], &tc.2, &[], &mut gx);
            assert_vector_ulps_eq!(tc.3, &gx.to_vec());
        }
    }

    #[test]
    fn check_batch_pick_bw_1n() {
        let a_data = vec![0., 1., 2., 3.];
        struct TestCase(Shape, Vec<f32>, Vec<u32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                vec![0, 0, 0],
                vec![6., 7., 8., 9.],
            ),
            TestCase(
                shape![2, 2; 2],
                vec![1., 1., 1., 1., 2., 2., 2., 2.],
                vec![0, 0],
                vec![3., 4., 5., 6.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2], &a_data);
            dev.call_bw_impl("batch_pick_bw_impl", &[], &[], &[&gy], &tc.2, &[], &mut gx);
            assert_vector_ulps_eq!(tc.3, &gx.to_vec());
        }
    }
}
