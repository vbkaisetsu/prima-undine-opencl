define_opencl_fw_const_impl!(PReLUFwImpl, prelu_fw_kernel);
define_opencl_bw_const_impl!(PReLUBwImpl, prelu_bw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_prelu_fw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0., 0.5, 1., 2., 4., 8., 0., -0.5, -1., -2., -4., -8.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        for &k in &ks {
            let y_data = x_data
                .iter()
                .map(|&x| if x > 0. { x } else { x * k })
                .collect::<Vec<f32>>();
            let mut y = dev.new_tensor(shape![2, 3; 2]);
            y.alloc();
            dev.call_fw_impl("prelu_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_prelu_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        for &k in &ks {
            let y_data = vec![0., 1., 2., 3., 0., -k, -2. * k, -3. * k];
            let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let gy = dev.new_tensor_by_slice(y.shape(), &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            let gx_data = gy_data
                .iter()
                .zip(&x_data)
                .map(|(&gy, &x)| 1. + gy * if x > 0. { 1. } else { k })
                .collect::<Vec<f32>>();
            dev.call_bw_impl("prelu_bw_impl", &[&x], &[&y], &[&gy], &[], &[k], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }
}
