use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(TransposeFwImpl, transpose_fw_kernel);
impl FunctionFwImpl for TransposeFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let rows = x.shape()[0];
        let cols = x.shape()[1];
        let bs = x.shape().batch() as usize;
        let g1 = super::common::calc_num_blocks(rows as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(cols as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&rows)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&cols)).unwrap();
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

define_opencl_impl_struct!(TransposeBwImpl, transpose_bw_kernel);
impl FunctionBwImpl for TransposeBwImpl {
    fn call(
        &self,
        _xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let gy = gys[0];
        let rows = gx.shape()[0];
        let cols = gx.shape()[1];
        let bs = gx.shape().batch() as usize;
        let g1 = super::common::calc_num_blocks(rows as usize, self.wgs[0]);
        let g2 = super::common::calc_num_blocks(cols as usize, self.wgs[1]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&rows)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&cols)).unwrap();
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
    fn check_transpose_fw_11() {
        let x_data = vec![42.];
        let y_data = vec![42.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![], &x_data);
        let mut y = dev.new_tensor(shape![]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_n1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![12], &x_data);
        let mut y = dev.new_tensor(shape![1, 12]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_1n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![1, 3; 4], &x_data);
        let mut y = dev.new_tensor(shape![3; 4]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_nn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 2., 4., 5., 7., 6., 8., 9., 11., 10., 12.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 3], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 3]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_mn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 2; 2]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_bw_11() {
        let gy_data = vec![42.];
        let gx_data = vec![43.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_n1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![12], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 12], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_1n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![1, 12], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![12], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_nn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 4., 3., 5., 6., 8., 7., 9., 10., 12., 11., 13.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![2, 2; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_mn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 4., 6., 3., 5., 7., 8., 10., 12., 9., 11., 13.];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![2, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 2; 2], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
