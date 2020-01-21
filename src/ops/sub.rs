define_opencl_fw_ab_impl!(SubFwImpl, sub_fw_kernel);
define_opencl_bw_a_impl!(SubBwAImpl, sub_bw_a_kernel);
define_opencl_bw_b_impl!(SubBwBImpl, sub_bw_b_kernel);
define_opencl_fw_const_impl!(SubConstLFwImpl, sub_const_l_fw_kernel);
define_opencl_bw_const_impl!(SubConstLBwImpl, sub_const_l_bw_kernel);
define_opencl_fw_const_impl!(SubConstRFwImpl, sub_const_r_fw_kernel);
define_opencl_bw_const_impl!(SubConstRBwImpl, sub_const_r_bw_kernel);
define_opencl_fw_ab_impl!(SubScalarLFwImpl, sub_scalar_l_fw_kernel);
define_opencl_fw_ab_impl!(SubScalarRFwImpl, sub_scalar_r_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_sub_const_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 1.;
        let y_data = vec![-999., -99., -9., 0., 0.9, 0.99, 0.999, 0.9999];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_const_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 1.;
        let y_data = vec![999., 99., 9., 0., -0.9, -0.99, -0.999, -0.9999];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 1.];
        let y_data = vec![-990., -90., 0., 9., 0.9, 0.99, 0.999, 0.9999];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 1.];
        let y_data = vec![990., 90., 0., -9., -0.9, -0.99, -0.999, -0.9999];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_l_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![1.];
            let y_data = vec![-999., -99., -9., 0., 0.9, 0.99, 0.999, 0.9999];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 1.];
            let y_data = vec![-990., -90., 0., 9., -999., -99., -9., 0.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_sub_scalar_r_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![1.];
            let y_data = vec![999., 99., 9., 0., -0.9, -0.99, -0.999, -0.9999];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 1.];
            let y_data = vec![990., 90., 0., -9., 999., 99., 9., 0.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_sub_fw() {
        let a_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let b_data = vec![0., 100., 20., 3., 0.4, 0.05, 0.006, 0.0007];
        let y1_data = vec![1000., 0., -10., -2., -0.3, -0.04, -0.005, -0.0006];
        let y2_data = vec![-1000., 0., 10., 2., 0.3, 0.04, 0.005, 0.0006];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_sub_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let y1_data = vec![0., 1., 2., 3., -4., -3., -2., -1.];
        let y2_data = vec![0., -1., -2., -3., 4., 3., 2., 1.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_sub_const_l_bw() {
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let gx_data = vec![0., 2., -1., 3., -1., 3., 0., 2.];
        let dev = get_device();
        let x = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
        let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
        dev.call_bw_impl(
            "sub_const_l_bw_impl",
            &[&x],
            &[&y],
            &[&gy],
            &[],
            &[std::f32::NAN],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_sub_const_r_bw() {
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let gx_data = vec![2., 0., 3., -1., 3., -1., 2., 0.];
        let dev = get_device();
        let x = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
        let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
        dev.call_bw_impl(
            "sub_const_r_bw_impl",
            &[&x],
            &[&y],
            &[&gy],
            &[],
            &[std::f32::NAN],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_sub_bw_11() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_1n() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![4., -2.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_n1() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![-2., 4.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_nn() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
