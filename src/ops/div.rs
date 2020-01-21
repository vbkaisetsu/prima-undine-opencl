define_opencl_fw_ab_impl!(DivFwImpl, div_fw_kernel);
define_opencl_bw_a_impl!(DivBwAImpl, div_bw_a_kernel);
define_opencl_bw_b_impl!(DivBwBImpl, div_bw_b_kernel);
define_opencl_fw_const_impl!(DivConstLFwImpl, div_const_l_fw_kernel);
define_opencl_bw_const_impl!(DivConstLBwImpl, div_const_l_bw_kernel);
define_opencl_fw_const_impl!(DivConstRFwImpl, div_const_r_fw_kernel);
define_opencl_bw_const_impl!(DivConstRBwImpl, div_const_r_bw_kernel);
define_opencl_fw_ab_impl!(DivScalarLFwImpl, div_scalar_l_fw_kernel);
define_opencl_fw_ab_impl!(DivScalarRFwImpl, div_scalar_r_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_div_const_l_fw() {
        let x_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let k = 10.;
        let y_data = vec![0.01, -0.1, 1., -10., 100., -1000., 10000., -10000.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_const_r_fw() {
        let x_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let k = 10.;
        let y_data = vec![100., -10., 1., -0.1, 0.01, -0.001, 0.0001, -0.0001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 0.1];
        let y_data = vec![0.01, 0.1, 1., 10., 1., 10., 100., 1000.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 0.1];
        let y_data = vec![100., 10., 1., 0.1, 1., 0.1, 0.01, 0.001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_l_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.001];
            let k_data = vec![10.];
            let y_data = vec![0.01, 0.1, 1., 10., 100., 1000., 10000., 10000.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 0.1];
            let y_data = vec![0.01, 0.1, 1., 10., 0.0001, 0.001, 0.01, 0.1];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_div_scalar_r_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.001];
            let k_data = vec![10.];
            let y_data = vec![100., 10., 1., 0.1, 0.01, 0.001, 0.0001, 0.0001];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 0.1];
            let y_data = vec![100., 10., 1., 0.1, 10000., 1000., 100., 10.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_div_fw() {
        let a_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let b_data = vec![1., 2., 3., 4., -5., -6., -7., -8.];
        let y1_data = vec![
            1000.,
            -50.,
            10. / 3.,
            -0.25,
            -0.02,
            0.01 / 6.,
            -0.001 / 7.,
            1.25e-4,
        ];
        let y2_data = vec![0.001, -0.02, 0.3, -4., -50., 600., -7000., 8000.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("div_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_div_fw_batch_broadcast() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 1., 1., 1., 1., 2., 3., 4.];
        let y1_data = vec![1., 2., 3., 4., 1., 1., 1., 1.];
        let y2_data = vec![1., 0.5, 1. / 3., 0.25, 1., 1., 1., 1.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("div_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_div_const_l_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0.1, 1., 2., 3., -0.1, -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = get_device();
        for k in ks {
            let gx_data = vec![
                1. - 100. * k,
                1. + k,
                1. - k / 2.,
                1. + 2. * k / 9.,
                1. - 200. * k,
                1. + 2. * k,
                1. - k / 4.,
                1. + k / 9.,
            ];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
            let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
            dev.call_bw_impl(
                "div_const_l_bw_impl",
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
    fn check_div_const_r_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = get_device();
        for k in ks {
            let gx_data = vec![
                1. + 1. / k,
                1. - 1. / k,
                1. + 2. / k,
                1. - 2. / k,
                1. + 2. / k,
                1. - 2. / k,
                1. + 1. / k,
                1. - 1. / k,
            ];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
            let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
            dev.call_bw_impl(
                "div_const_r_bw_impl",
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
    fn check_div_bw_11() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1.];
        let ga_data = vec![1.1, 0.];
        let gb_data = vec![0.99, 11.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_1n() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![0.9, 2.];
        let gb_data = vec![0.99, 11., 0.98, 21.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_n1() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![1.1, 0., 1.2, -1.];
        let gb_data = vec![1.01, -9.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_nn() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![1.1, 0., 0.8, 3.];
        let gb_data = vec![0.99, 11., 1.02, -19.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
