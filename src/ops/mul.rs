define_opencl_fw_ab_impl!(MulFwImpl, mul_fw_kernel);
define_opencl_bw_a_impl!(MulBwAImpl, mul_bw_a_kernel);
define_opencl_bw_b_impl!(MulBwBImpl, mul_bw_b_kernel);
define_opencl_fw_const_impl!(MulConstFwImpl, mul_const_fw_kernel);
define_opencl_bw_const_impl!(MulConstBwImpl, mul_const_bw_kernel);
define_opencl_fw_ab_impl!(MulScalarFwImpl, mul_scalar_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_mul_const_fw() {
        let x_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.0001];
        let k = 10.;
        let y_data = vec![10000., -1000., 100., -10., 1., -0.1, 0.01, -0.001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_const_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_mul_scalar_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![0.1, 10.];
        let y_data = vec![100., 10., 1., 0.1, 1., 0.1, 0.01, 0.001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_mul_scalar_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![10.];
            let y_data = vec![10000., 1000., 100., 10., 1., 0.1, 0.01, 0.001];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("mul_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![0.1, 10.];
            let y_data = vec![100., 10., 1., 0.1, 10000., 1000., 100., 10.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("mul_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_mul_fw() {
        let a_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.0001];
        let b_data = vec![0., 1., 2., 3., -4., -5., -6., -7.];
        let y_data = vec![0., -100., 20., -3., -0.4, 0.05, -0.006, 0.0007];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_mul_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![1., 1., 1., 1., 0., 1., 2., 3.];
        let y_data = vec![0., 1., 2., 3., 0., 1., 4., 9.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        {
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("mul_fw_impl", &[&b, &a], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_mul_const_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = get_device();
        for k in ks {
            let gx_data = vec![
                1. + k,
                1. - k,
                1. + 2. * k,
                1. - 2. * k,
                1. + 2. * k,
                1. - 2. * k,
                1. + k,
                1. - k,
            ];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("mul_const_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
            let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
            dev.call_bw_impl(
                "mul_const_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[],
                &[k],
                &mut gx,
            );
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_mul_bw_11() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1.];
        let ga_data = vec![11., 0.];
        let gb_data = vec![2., -9.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2]);
        y.alloc();
        dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("mul_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("mul_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_mul_bw_1n() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![-9., 2.];
        let gb_data = vec![2., -9., 3., -19.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("mul_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("mul_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_mul_bw_n1() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![11., 0., 21., -1.];
        let gb_data = vec![0., 11.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("mul_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("mul_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_mul_bw_nn() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![11., 0., -19., 3.];
        let gb_data = vec![2., -9., -1., 21.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("mul_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("mul_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
