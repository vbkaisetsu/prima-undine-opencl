define_opencl_fw_ab_impl!(PowfFwImpl, powf_fw_kernel);
define_opencl_bw_a_impl!(PowfBwAImpl, powf_bw_a_kernel);
define_opencl_bw_b_impl!(PowfBwBImpl, powf_bw_b_kernel);
define_opencl_fw_const_impl!(PowfConstLFwImpl, powf_const_l_fw_kernel);
define_opencl_bw_const_impl!(PowfConstLBwImpl, powf_const_l_bw_kernel);
define_opencl_fw_const_impl!(PowfConstRFwImpl, powf_const_r_fw_kernel);
define_opencl_bw_const_impl!(PowfConstRBwImpl, powf_const_r_bw_kernel);
define_opencl_fw_ab_impl!(PowfScalarLFwImpl, powf_scalar_l_fw_kernel);
define_opencl_fw_ab_impl!(PowfScalarRFwImpl, powf_scalar_r_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_powf_const_r_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k = 3.;
        let y_data = vec![1., 8., 27., 64., 125., 216., 343., 512.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_const_l_fw() {
        let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let k = 3.;
        let y_data = vec![27., 9., 3., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_r_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k_data = vec![3., -3.];
        let y_data = vec![1., 8., 27., 64., 1. / 125., 1. / 216., 1. / 343., 1. / 512.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_l_fw() {
        let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let k_data = vec![2., 3.];
        let y_data = vec![8., 4., 2., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_r_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
            let k_data = vec![3.];
            let y_data = vec![1., 8., 27., 64., 125., 216., 343., 512.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1., 2., 3., 4.];
            let k_data = vec![3., -3.];
            let y_data = vec![1., 8., 27., 64., 1., 1. / 8., 1. / 27., 1. / 64.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powf_const_l_fw_batch_broadcast() {
        let dev = get_device();
        {
            let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
            let k_data = vec![3.];
            let y_data = vec![27., 9., 3., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![3., 2., 1., 0.];
            let k_data = vec![2., 3.];
            let y_data = vec![8., 4., 2., 1., 27., 9., 3., 1.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powf_fw() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3.];
        let b_data = vec![2., 2., 2., 2., 3., 3., 3., 3.];
        let y1_data = vec![0., 1., 4., 9., 0., 1., 8., 27.];
        let y2_data = vec![1., 2., 4., 8., 1., 3., 9., 27.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_powf_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![2., 2., 2., 2., 3., 3., 3., 3.];
        let y1_data = vec![0., 1., 4., 9., 0., 1., 8., 27.];
        let y2_data = vec![1., 2., 4., 8., 1., 3., 9., 27.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_powf_const_r_bw() {
        let x_data = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let ks = vec![1., 0.5, 0.25, 0. - 0.125, -0.25, -0.5, -0.1];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &k in &ks {
            let y_data = x_data.iter().map(|&x| x.powf(k)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * k * x.powf(k - 1.))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powf_const_r_bw_impl",
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
    fn check_powf_const_l_bw() {
        let x_data = vec![1., 0.5, 0.25, 0., -0.125, -0.25, -0.5, -1.];
        let ks: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &k in &ks {
            let y_data = x_data.iter().map(|&x| k.powf(x)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * k.ln() * k.powf(x))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powf_const_l_bw_impl",
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
    fn check_powf_bw_11() {
        let a_data = vec![1., 2., 4., 8.];
        let b_data = vec![2., 1., 0., -1.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let y_data = a_data
            .iter()
            .zip(&b_data)
            .map(|(&a, &b)| a.powf(b))
            .collect::<Vec<f32>>();
        let y = dev.new_tensor_by_slice(shape![2, 2], &y_data);
        let gy_data = vec![1., -1., 2., -2.];
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let ga_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * b * a.powf(b - 1.))
            .collect::<Vec<f32>>();
        let gb_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * a.ln() * a.powf(b))
            .collect::<Vec<f32>>();
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_1n() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8.];
        let b_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let mut y_data = vec![0.; 8];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let mut ga_data = vec![1.; 4];
        let mut gb_data = vec![1.; 8];
        for ib in 0..b_data.len() {
            let ia = ib % a_data.len();
            y_data[ib] = a_data[ia].powf(b_data[ib]);
            ga_data[ia] += gy_data[ib] * b_data[ib] * a_data[ia].powf(b_data[ib] - 1.);
            gb_data[ib] += gy_data[ib] * a_data[ia].ln() * a_data[ia].powf(b_data[ib]);
        }
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_n1() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let b_data = vec![2., 1., 0., -1.];
        let mut y_data = vec![0.; 8];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let mut ga_data = vec![1.; 8];
        let mut gb_data = vec![1.; 4];
        for ia in 0..a_data.len() {
            let ib = ia % b_data.len();
            y_data[ia] = a_data[ia].powf(b_data[ib]);
            ga_data[ia] += gy_data[ia] * b_data[ib] * a_data[ia].powf(b_data[ib] - 1.);
            gb_data[ib] += gy_data[ia] * a_data[ia].ln() * a_data[ia].powf(b_data[ib]);
        }
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_nn() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let b_data = vec![1., 0.5, 0.25, 0., -0.125, -0.25, -0.5, -1.];
        let y_data = a_data
            .iter()
            .zip(&b_data)
            .map(|(&a, &b)| a.powf(b))
            .collect::<Vec<f32>>();
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let ga_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * b * a.powf(b - 1.))
            .collect::<Vec<f32>>();
        let gb_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * a.ln() * a.powf(b))
            .collect::<Vec<f32>>();
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
