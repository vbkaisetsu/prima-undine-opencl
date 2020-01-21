use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(FlipFwImpl, flip_fw_kernel);
impl FunctionFwImpl for FlipFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape()[dim];
        let skip = x.shape().lower_volume(dim);
        let r = x.shape().size() / n;
        let g1 = super::common::calc_num_blocks(n as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(r as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&skip)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&n)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&r)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                2,
                None,
                &[g1 * self.wgs[0], g2 * self.wgs[1], 1],
                Some([self.wgs[0], self.wgs[1], 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

define_opencl_impl_struct!(FlipBwImpl, flip_bw_kernel);
impl FunctionBwImpl for FlipBwImpl {
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
        let n = gx.shape()[dim];
        let skip = gx.shape().lower_volume(dim);
        let r = gx.shape().size() / n;
        let g1 = super::common::calc_num_blocks(n as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(r as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&skip)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&n)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&r)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(buffer!(gx))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                2,
                None,
                &[g1 * self.wgs[0], g2 * self.wgs[1], 1],
                Some([self.wgs[0], self.wgs[1], 1]),
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
    fn check_flip_01_fw() {
        let x_data = vec![42.];
        let y_data = vec![vec![42.], vec![42.], vec![42.], vec![42.]];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape![]);
            y.alloc();
            dev.call_fw_impl("flip_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_flip_11_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![
            vec![12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![12], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape![12]);
            y.alloc();
            dev.call_fw_impl("flip_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_flip_21_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![
            vec![6., 5., 4., 3., 2., 1., 12., 11., 10., 9., 8., 7.],
            vec![7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![6, 2], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape![6, 2]);
            y.alloc();
            dev.call_fw_impl("flip_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_flip_31_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![
            vec![3., 2., 1., 6., 5., 4., 9., 8., 7., 12., 11., 10.],
            vec![4., 5., 6., 1., 2., 3., 10., 11., 12., 7., 8., 9.],
            vec![7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6.],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 2, 2], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape![3, 2, 2]);
            y.alloc();
            dev.call_fw_impl("flip_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_flip_32_fw() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let y_data = vec![
            vec![
                3., 2., 1., 6., 5., 4., 9., 8., 7., 12., 11., 10., 15., 14., 13., 18., 17., 16.,
                21., 20., 19., 24., 23., 22.,
            ],
            vec![
                4., 5., 6., 1., 2., 3., 10., 11., 12., 7., 8., 9., 16., 17., 18., 13., 14., 15.,
                22., 23., 24., 19., 20., 21.,
            ],
            vec![
                7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 19., 20., 21., 22., 23., 24.,
                13., 14., 15., 16., 17., 18.,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 2, 2; 2], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape![3, 2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("flip_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_flip_bw() {
        let gy_data = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23.,
        ];
        let gx_data = vec![
            vec![
                3., 2., 1., 6., 5., 4., 9., 8., 7., 12., 11., 10., 15., 14., 13., 18., 17., 16.,
                21., 20., 19., 24., 23., 22.,
            ],
            vec![
                4., 5., 6., 1., 2., 3., 10., 11., 12., 7., 8., 9., 16., 17., 18., 13., 14., 15.,
                22., 23., 24., 19., 20., 21.,
            ],
            vec![
                7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 19., 20., 21., 22., 23., 24.,
                13., 14., 15., 16., 17., 18.,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        ];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 2, 2; 2], &gy_data);
        for i in 0..4 {
            let mut gx = dev.new_tensor_by_constant(shape![3, 2, 2; 2], 1.);
            dev.call_bw_impl("flip_bw_impl", &[], &[], &[&gy], &[i], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data[i as usize], gx.to_vec());
        }
    }
}
