use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(ConcatFwImpl, concat_fw_kernel);
impl FunctionFwImpl for ConcatFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let dim = u32data[0];
        let y = &mut ys[0];
        let new_bs = y.shape().batch();
        let base = y.shape().lower_volume(dim);
        let skip = base * y.shape()[dim];
        let repeat = y.shape().volume() / skip;
        let mut offset = 0;
        for x in xs {
            let span = base * x.shape()[dim];
            let x_size = span * repeat * x.shape().batch();
            let y_size = span * repeat * new_bs;
            let g1 = super::common::calc_num_blocks(y_size as usize, self.wgs[0]);
            let queue = &self.internal.queue;
            let kernel = self.kernel.lock().unwrap();
            unsafe {
                ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&span)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&skip)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&x_size)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&y_size)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 5, ArgVal::mem(buffer!(y))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&offset)).unwrap();
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
            offset += span;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_concat_fw_n_3x3() {
        let a_data = vec![1., 1., 1.];
        let b_data = vec![2., 3., 2., 3., 2., 3.];
        let c_data = vec![4., 5., 6., 4., 5., 6., 4., 5., 6.];
        let y_data = vec![
            1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![1, 3], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 3], &b_data);
        let c = dev.new_tensor_by_slice(shape![3, 3], &c_data);
        let mut y = dev.new_tensor(shape![6, 3]);
        y.alloc();
        dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_concat_fw_5x4() {
        let shapes = vec![shape![20], shape![5, 4], shape![5, 1, 4]];
        let y_data = vec![
            1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![5], 1.);
        let b = dev.new_tensor_by_constant(shape![5], 2.);
        let c = dev.new_tensor_by_constant(shape![5], 3.);
        let d = dev.new_tensor_by_constant(shape![5], 4.);
        for &i in &[0, 1, 2] {
            let mut y = dev.new_tensor(shapes[i]);
            y.alloc();
            dev.call_fw_impl(
                "concat_fw_impl",
                &[&a, &b, &c, &d],
                &[i as u32],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_concat_fw_2_2_2x2() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 11., 22., 33., 44., 55., 66., 77., 88.,
        ];
        let b_data = vec![
            -1., -2., -3., -4., -5., -6., -7., -8., -11., -22., -33., -44., -55., -66., -77., -88.,
        ];
        let shapes = vec![
            shape![4, 2, 2; 2],
            shape![2, 4, 2; 2],
            shape![2, 2, 4; 2],
            shape![2, 2, 2, 2; 2],
            shape![2, 2, 2, 1, 2; 2],
        ];
        let y_data = vec![
            vec![
                1., 2., -1., -2., 3., 4., -3., -4., 5., 6., -5., -6., 7., 8., -7., -8., 11., 22.,
                -11., -22., 33., 44., -33., -44., 55., 66., -55., -66., 77., 88., -77., -88.,
            ],
            vec![
                1., 2., 3., 4., -1., -2., -3., -4., 5., 6., 7., 8., -5., -6., -7., -8., 11., 22.,
                33., 44., -11., -22., -33., -44., 55., 66., 77., 88., -55., -66., -77., -88.,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8., 11., 22.,
                33., 44., 55., 66., 77., 88., -11., -22., -33., -44., -55., -66., -77., -88.,
            ],
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &b_data);
        for &i in &[0, 1, 2, 3, 4] {
            let mut y = dev.new_tensor(shapes[i]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b], &[i as u32], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[if i < 2 { i } else { 2 }], y.to_vec());
        }
    }

    #[test]
    fn check_concat_fw_batch_broadcast() {
        let dev = get_device();
        {
            let a_data = vec![1., 1., 11., 11.];
            let b_data = vec![2., 2., 2., 2.];
            let c_data = vec![3., 3., 3., 3., 3., 3.];
            let y_data = vec![
                1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 11., 11., 2., 2., 2., 2., 3., 3.,
                3., 3., 3., 3.,
            ];
            let a = dev.new_tensor_by_slice(shape![2, 1; 2], &a_data);
            let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
            let c = dev.new_tensor_by_slice(shape![2, 3], &c_data);
            let mut y = dev.new_tensor(shape![2, 6; 2]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[1], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let a_data = vec![1., 1., 1., 1., 1., 1.];
            let b_data = vec![2., 2., 2., 2., 22., 22., 22., 22.];
            let c_data = vec![3., 3., 33., 33.];
            let y_data = vec![
                1., 1., 1., 2., 2., 3., 1., 1., 1., 2., 2., 3., 1., 1., 1., 22., 22., 33., 1., 1.,
                1., 22., 22., 33.,
            ];
            let a = dev.new_tensor_by_slice(shape![3, 2], &a_data);
            let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            let c = dev.new_tensor_by_slice(shape![1, 2; 2], &c_data);
            let mut y = dev.new_tensor(shape![6, 2; 2]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let a_data = vec![1.];
            let b_data = vec![2.];
            let c_data = vec![3., 33., 333.];
            let y_data = vec![1., 2., 3., 1., 2., 33., 1., 2., 333.];
            let a = dev.new_tensor_by_slice(shape![], &a_data);
            let b = dev.new_tensor_by_slice(shape![], &b_data);
            let c = dev.new_tensor_by_slice(shape![; 3], &c_data);
            let mut y = dev.new_tensor(shape![3; 3]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }
}
