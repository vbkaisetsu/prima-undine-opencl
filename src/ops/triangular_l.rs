use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(TriangularLFwImpl, triangular_l_fw_kernel);
impl FunctionFwImpl for TriangularLFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let k = u32data[0];
        let size = x.shape()[0];
        let bs = x.shape().batch() as usize;
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(size as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&k)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                3,
                None,
                &[g1 * self.wgs[0], g2 * self.wgs[1], bs],
                Some([self.wgs[0], self.wgs[1], 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

define_opencl_impl_struct!(TriangularLBwImpl, triangular_l_bw_kernel);
impl FunctionBwImpl for TriangularLBwImpl {
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
        let k = u32data[0];
        let size = gx.shape()[0];
        let bs = gx.shape().batch() as usize;
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(size as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&k)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(buffer!(gx))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
                &kernel,
                3,
                None,
                &[g1 * self.wgs[0], g2 * self.wgs[1], bs],
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
    fn check_triangular_l_fw_11() {
        let x_data = vec![42.];
        let y_data = vec![42.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![], &x_data);
        let mut y = dev.new_tensor(shape![]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_fw_nn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![1., 2., 3., 0., 5., 6., 0., 0., 9.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_fw_nn_k1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![0., 2., 3., 0., 0., 6., 0., 0., 0.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[1], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_fw_nn_k2() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![0., 0., 3., 0., 0., 0., 0., 0., 0.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[2], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_fw_batch_nn() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let y_data = vec![
            1., 2., 3., 0., 5., 6., 0., 0., 9., 10., 11., 12., 0., 14., 15., 0., 0., 18.,
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 3; 2]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_fw_batch_nn_k1() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let y_data = vec![
            0., 2., 3., 0., 0., 6., 0., 0., 0., 0., 11., 12., 0., 0., 15., 0., 0., 0.,
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 3; 2]);
        y.alloc();
        dev.call_fw_impl("triangular_l_fw_impl", &[&x], &[1], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_11() {
        let gy_data = vec![42.];
        let gx_data = vec![43.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_nn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![2., 3., 4., 1., 6., 7., 1., 1., 10.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_nn_k1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![1., 3., 4., 1., 1., 7., 1., 1., 1.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[1], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_nn_k2() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![1., 1., 4., 1., 1., 1., 1., 1., 1.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[2], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_batch_nn() {
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let gx_data = vec![
            2., 3., 4., 1., 6., 7., 1., 1., 10., 11., 12., 13., 1., 15., 16., 1., 1., 19.,
        ];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3; 2], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_l_bw_batch_nn_k1() {
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let gx_data = vec![
            1., 3., 4., 1., 1., 7., 1., 1., 1., 1., 12., 13., 1., 1., 16., 1., 1., 1.,
        ];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3; 2], 1.);
        dev.call_bw_impl("triangular_l_bw_impl", &[], &[], &[&gy], &[1], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
