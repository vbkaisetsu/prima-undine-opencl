use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(PickFwImpl, pick_fw_kernel);
impl FunctionFwImpl for PickFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let ids = &u32data[1..];
        let y = &mut ys[0];
        let wy = y.shape().lower_volume(dim);
        let wx = wy * x.shape()[dim];
        let sx = if x.shape().has_batch() {
            x.shape().volume()
        } else {
            0
        };
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
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&wx)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&wy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&sx)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&si)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&sy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 7, ArgVal::mem(buffer!(y))).unwrap();
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

define_opencl_impl_struct!(PickBwImpl, pick_bw_kernel);
impl FunctionBwImpl for PickBwImpl {
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
        let dim = u32data[0];
        let ids = &u32data[1..];
        let wy = gy.shape().lower_volume(dim);
        let wx = wy * gx.shape()[dim];
        let sx = if gx.shape().has_batch() {
            gx.shape().volume()
        } else {
            0
        };
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
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&wx)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&wy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&sx)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&si)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&sy)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 7, ArgVal::mem(buffer!(gx))).unwrap();
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
    fn check_pick_fw_nn() {
        struct TestCase(Shape, u32, Vec<u32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![0, 0, 0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![1, 0, 1],
                shape![1, 2, 2; 3],
                vec![1., 3., 5., 7., 8., 10., 12., 14., 17., 19., 21., 23.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.],
            ),
            TestCase(
                shape![2, 2, 2],
                0,
                vec![0, 1, 0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 1., 3., 5., 7., 0., 2., 4., 6.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                1,
                vec![0, 0, 0],
                shape![2, 1, 2; 3],
                vec![0., 1., 4., 5., 8., 9., 12., 13., 16., 17., 20., 21.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                2,
                vec![0, 0, 0],
                shape![2, 2, 1; 3],
                vec![0., 1., 2., 3., 8., 9., 10., 11., 16., 17., 18., 19.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x_data = (0..tc.0.size()).map(|x| x as f32).collect::<Vec<f32>>();
            let x = dev.new_tensor_by_slice(tc.0, &x_data);
            let mut y = dev.new_tensor(tc.3);
            y.alloc();
            let mut u32data = vec![tc.1];
            u32data.append(&mut tc.2.clone());
            dev.call_fw_impl("pick_fw_impl", &[&x], &u32data, &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.4, y.to_vec());
        }
    }

    #[test]
    fn check_pick_bw_n1() {
        struct TestCase(Shape, Vec<f32>, u32, Vec<u32>, Vec<f32>);
        let gx_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0],
                vec![1., 1., 3., 3., 2., 1., 4., 3., 3., 1., 5., 3.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1],
                vec![0., 2., 2., 4., 0., 3., 2., 5., 0., 4., 2., 6.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0],
                vec![1., 2., 2., 3., 2., 3., 2., 3., 3., 4., 2., 3.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1],
                vec![0., 1., 3., 4., 0., 1., 4., 5., 0., 1., 5., 6.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                2,
                vec![0],
                vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &gx_data);
            let mut u32data = vec![tc.2];
            u32data.append(&mut tc.3.clone());
            dev.call_bw_impl("pick_bw_impl", &[], &[], &[&gy], &u32data, &[], &mut gx);
            assert_vector_ulps_eq!(tc.4, gx.to_vec());
        }
    }

    #[test]
    fn check_pick_bw_1n() {
        struct TestCase(Shape, Vec<f32>, u32, Vec<u32>, Vec<f32>);
        let gx_data = vec![0., 1., 2., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0, 0, 0],
                vec![6., 1., 8., 3.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1, 1, 1],
                vec![0., 7., 2., 9.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0, 1, 0],
                vec![4., 3., 6., 5.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1, 0, 1],
                vec![2., 5., 4., 7.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0, 0, 0],
                vec![6., 7., 2., 3.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1, 1, 1],
                vec![0., 1., 8., 9.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0, 1, 0],
                vec![4., 5., 4., 5.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1, 0, 1],
                vec![2., 3., 6., 7.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                2,
                vec![0, 0, 0],
                vec![6., 7., 8., 9.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2], &gx_data);
            let mut u32data = vec![tc.2];
            u32data.append(&mut tc.3.clone());
            dev.call_bw_impl("pick_bw_impl", &[], &[], &[&gy], &u32data, &[], &mut gx);
            assert_vector_ulps_eq!(tc.4, gx.to_vec());
        }
    }
}
