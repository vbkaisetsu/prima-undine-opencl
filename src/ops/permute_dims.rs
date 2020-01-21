use std::cmp;

use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(PermuteDimsFwImpl, permute_dims_fw_kernel);
impl FunctionFwImpl for PermuteDimsFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let perm = u32data;
        let y = &mut ys[0];
        let ndims = perm.len();
        let bs = x.shape().batch() as usize;
        let size = x.shape().volume();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let mut x_strides = vec![0; ndims];
        let mut y_strides = vec![0; ndims];
        let mut x_stride_tmp = 1;
        let mut y_stride_tmp = 1;
        for i in 0..ndims {
            x_strides[ndims - i - 1] = x_stride_tmp;
            y_strides[ndims - perm[i] as usize - 1] = y_stride_tmp;
            x_stride_tmp *= x.shape()[i as u32];
            y_stride_tmp *= y.shape()[i as u32];
        }
        let x_stride_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                cmp::max(x_strides.len(), 1),
                None::<&[u32]>,
            )
            .unwrap()
        };
        let y_stride_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                cmp::max(y_strides.len(), 1),
                None::<&[u32]>,
            )
            .unwrap()
        };
        if perm.len() != 0 {
            unsafe {
                super::common::write_buffer(&self.internal.queue, &x_strides, &x_stride_buf);
                super::common::write_buffer(&self.internal.queue, &y_strides, &y_stride_buf);
            }
        }
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&(ndims as u32))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(&x_stride_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(&y_stride_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 5, ArgVal::mem(buffer!(y))).unwrap();
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

define_opencl_impl_struct!(PermuteDimsBwImpl, permute_dims_bw_kernel);
impl FunctionBwImpl for PermuteDimsBwImpl {
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
        let perm = u32data;
        let ndims = perm.len();
        let bs = gx.shape().batch() as usize;
        let size = gx.shape().volume();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let mut x_strides = vec![0; ndims];
        let mut y_strides = vec![0; ndims];
        let mut x_stride_tmp = 1;
        let mut y_stride_tmp = 1;
        for i in 0..ndims {
            x_strides[ndims - i - 1] = x_stride_tmp;
            y_strides[ndims - perm[i] as usize - 1] = y_stride_tmp;
            x_stride_tmp *= gx.shape()[i as u32];
            y_stride_tmp *= gy.shape()[i as u32];
        }
        let x_stride_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                cmp::max(x_strides.len(), 1),
                None::<&[u32]>,
            )
            .unwrap()
        };
        let y_stride_buf = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                cmp::max(y_strides.len(), 1),
                None::<&[u32]>,
            )
            .unwrap()
        };
        if perm.len() != 0 {
            unsafe {
                super::common::write_buffer(&self.internal.queue, &x_strides, &x_stride_buf);
                super::common::write_buffer(&self.internal.queue, &y_strides, &y_stride_buf);
            }
        }
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&(ndims as u32))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(&x_stride_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(&y_stride_buf)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 5, ArgVal::mem(buffer!(gx))).unwrap();
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

    #[test]
    fn check_permute_dims_fw_111() {
        let x_data = vec![42., 43.];
        let y_data = vec![42., 43.];
        let perm = vec![];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![; 2], &x_data);
        let mut y = dev.new_tensor(shape![; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_n11() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![1, 2, 0];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![6; 2], &x_data);
        let mut y = dev.new_tensor(shape![1, 1, 6; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_1n1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![0, 2, 1];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![1, 4; 3], &x_data);
        let mut y = dev.new_tensor(shape![1, 1, 4; 3]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_11n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &x_data);
        let mut y = dev.new_tensor(shape![4; 3]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_mn1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let perm = vec![1, 2, 0];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 1, 2; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_m1n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![0, 2, 1];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 1, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 2; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_1mn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![1, 3, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 1, 3; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_lmn() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let y_data = vec![
            1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12., 13., 19., 14., 20., 15., 21., 16.,
            22., 17., 23., 18., 24.,
        ];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_111() {
        let gy_data = vec![42., 43.];
        let gx_data = vec![43., 44.];
        let perm = vec![];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_n11() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![1, 0];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![1, 6; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![6; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_1n1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![0, 2, 1];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 4; 3], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_11n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![4; 3], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_mn1() {
        let gy_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![1, 2, 0];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 1, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 3; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_m1n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![0, 2, 1];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![3, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 1, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_1mn() {
        let gy_data = vec![1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![2, 1, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 3, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_lmn() {
        let gy_data = vec![
            1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12., 13., 19., 14., 20., 15., 21., 16.,
            22., 17., 23., 18., 24.,
        ];
        let gx_data = vec![
            2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
            21., 22., 23., 24., 25.,
        ];
        let perm = vec![2, 0, 1];
        let dev = get_device();
        let gy = dev.new_tensor_by_slice(shape![2, 2, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 3, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
