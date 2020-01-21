define_opencl_fw_ab_impl!(AddFwImpl, add_fw_kernel);
define_opencl_bw_a_impl!(AddBwAImpl, add_bw_a_kernel);
define_opencl_bw_b_impl!(AddBwBImpl, add_bw_b_kernel);
define_opencl_fw_const_impl!(AddConstFwImpl, add_const_fw_kernel);
define_opencl_bw_const_impl!(AddConstBwImpl, add_const_bw_kernel);
define_opencl_fw_ab_impl!(AddScalarFwImpl, add_scalar_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_add_const_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 1.;
        let y_data = vec![1001., 101., 11., 2., 1.1, 1.01, 1.001, 1.0001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("add_const_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_add_scalar_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 1.];
        let y_data = vec![1010., 110., 20., 11., 1.1, 1.01, 1.001, 1.0001];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("add_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_add_scalar_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![1.];
            let y_data = vec![1001., 101., 11., 2., 1.1, 1.01, 1.001, 1.0001];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("add_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 1.];
            let y_data = vec![1010., 110., 20., 11., 1001., 101., 11., 2.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("add_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_add_fw() {
        let a_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let b_data = vec![0., 100., 20., 3., 0.4, 0.05, 0.006, 0.0007];
        let y_data = vec![1000., 200., 30., 4., 0.5, 0.06, 0.007, 0.0008];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("add_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_add_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let y_data = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        {
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("add_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("add_fw_impl", &[&b, &a], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_add_const_bw() {
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = get_device();
        let gx_data = vec![2., 0., 3., -1., 3., -1., 2., 0.];
        let x = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
        let mut gx = dev.new_tensor_by_constant(x.shape(), 1.);
        dev.call_bw_impl(
            "add_const_bw_impl",
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
    fn check_add_bw_11() {
        let gy_data = vec![1., -1.];
        let ga_data = vec![2., 0.];
        let gb_data = vec![2., 0.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("add_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("add_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_add_bw_1n() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![4., -2.];
        let gb_data = vec![2., 0., 3., -1.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("add_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("add_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_add_bw_n1() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![4., -2.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("add_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("add_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_add_bw_nn() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![2., 0., 3., -1.];
        let dev = get_device();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("add_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("add_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
