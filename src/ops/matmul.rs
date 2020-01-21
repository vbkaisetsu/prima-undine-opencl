use std::ptr;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

use crate::clblast;

define_empty_impl!(MatmulFwImpl);
impl FunctionFwImpl for MatmulFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let a = xs[0];
        let b = xs[1];
        let y = &mut ys[0];
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![0.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            unsafe {
                clblast::CLBlastSgemmBatched(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::NO,
                    di,
                    dk,
                    dj,
                    alphas.as_ptr(),
                    buffer!(a).as_ptr(),
                    a_offsets.as_ptr(),
                    di,
                    buffer!(b).as_ptr(),
                    b_offsets.as_ptr(),
                    dj,
                    betas.as_ptr(),
                    buffer!(y).as_ptr(),
                    y_offsets.as_ptr(),
                    di,
                    bs,
                    &mut self.internal.queue.as_ptr(),
                    ptr::null_mut(),
                );
            }
        } else {
            let alpha = 1.;
            let beta = 0.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::NO,
                    di,
                    dk * b.shape().batch() as usize,
                    dj,
                    alpha,
                    buffer!(a).as_ptr(),
                    0,
                    di,
                    buffer!(b).as_ptr(),
                    0,
                    dj,
                    beta,
                    buffer!(y).as_ptr(),
                    0,
                    di,
                    &mut self.internal.queue.as_ptr(),
                    ptr::null_mut(),
                );
            }
        }
    }
}

define_empty_impl!(MatmulBwAImpl);
impl FunctionBwImpl for MatmulBwAImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let b = xs[1];
        let gy = gys[0];
        let ga = gx;
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![1.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            unsafe {
                clblast::CLBlastSgemmBatched(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::YES,
                    di,
                    dj,
                    dk,
                    alphas.as_ptr(),
                    buffer!(gy).as_ptr(),
                    y_offsets.as_ptr(),
                    di,
                    buffer!(b).as_ptr(),
                    b_offsets.as_ptr(),
                    dj,
                    betas.as_ptr(),
                    buffer!(ga).as_ptr(),
                    a_offsets.as_ptr(),
                    di,
                    bs,
                    &mut self.internal.queue.as_ptr(),
                    ptr::null_mut(),
                );
            }
        } else {
            let alpha = 1.;
            let beta = 1.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::NO,
                    clblast::transpose::YES,
                    di,
                    dj,
                    dk * b.shape().batch() as usize,
                    alpha,
                    buffer!(gy).as_ptr(),
                    0,
                    di,
                    buffer!(b).as_ptr(),
                    0,
                    dj,
                    beta,
                    buffer!(ga).as_ptr(),
                    0,
                    di,
                    &mut self.internal.queue.as_ptr(),
                    ptr::null_mut(),
                );
            }
        }
    }
}

define_empty_impl!(MatmulBwBImpl);
impl FunctionBwImpl for MatmulBwBImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let b = xs[1];
        let gy = gys[0];
        let gb = gx;
        let di = a.shape()[0] as usize;
        let dj = a.shape()[1] as usize;
        let dk = b.shape()[1] as usize;
        if a.shape().has_batch() {
            let a_skip = di * dj;
            let b_skip = if b.shape().has_batch() { dj * dk } else { 0 };
            let y_skip = di * dk;
            let bs = a.shape().batch() as usize;
            let alphas = vec![1.; bs];
            let betas = vec![1.; bs];
            let mut a_offsets = vec![0; bs];
            let mut b_offsets = vec![0; bs];
            let mut y_offsets = vec![0; bs];
            for n in 0..bs {
                a_offsets[n] = n * a_skip;
                b_offsets[n] = n * b_skip;
                y_offsets[n] = n * y_skip;
            }
            if b_skip > 0 {
                unsafe {
                    clblast::CLBlastSgemmBatched(
                        clblast::layout::COL_MAJOR,
                        clblast::transpose::YES,
                        clblast::transpose::NO,
                        dj,
                        dk,
                        di,
                        alphas.as_ptr(),
                        buffer!(a).as_ptr(),
                        a_offsets.as_ptr(),
                        di,
                        buffer!(gy).as_ptr(),
                        y_offsets.as_ptr(),
                        di,
                        betas.as_ptr(),
                        buffer!(gb).as_ptr(),
                        b_offsets.as_ptr(),
                        dj,
                        bs,
                        &mut self.internal.queue.as_ptr(),
                        ptr::null_mut(),
                    );
                }
            } else {
                let alpha = 1.;
                let beta = 1.;
                for n in 0..bs {
                    unsafe {
                        clblast::CLBlastSgemm(
                            clblast::layout::COL_MAJOR,
                            clblast::transpose::YES,
                            clblast::transpose::NO,
                            dj,
                            dk,
                            di,
                            alpha,
                            buffer!(a).as_ptr(),
                            n * a_skip,
                            di,
                            buffer!(gy).as_ptr(),
                            n * y_skip,
                            di,
                            beta,
                            buffer!(gb).as_ptr(),
                            n * b_skip,
                            dj,
                            &mut self.internal.queue.as_ptr(),
                            ptr::null_mut(),
                        );
                    }
                }
            }
        } else {
            let alpha = 1.;
            let beta = 1.;
            unsafe {
                clblast::CLBlastSgemm(
                    clblast::layout::COL_MAJOR,
                    clblast::transpose::YES,
                    clblast::transpose::NO,
                    dj,
                    dk * b.shape().batch() as usize,
                    di,
                    alpha,
                    buffer!(a).as_ptr(),
                    0,
                    di,
                    buffer!(gy).as_ptr(),
                    0,
                    di,
                    beta,
                    buffer!(gb).as_ptr(),
                    0,
                    dj,
                    &mut self.internal.queue.as_ptr(),
                    ptr::null_mut(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_matmul_fw_aa() {
        let x_data = vec![1., 2., 3., 4., 1., 0., 0., 1., 0., 2., 3., 0.];
        let y_data = vec![7., 10., 15., 22., 1., 0., 0., 1., 6., 0., 0., 6.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 3], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 3]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&x, &x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_ab() {
        let a_data = vec![
            1., 1000., 1., 10., 100., 10., 100., 10., 100., 1000., 1., 1000.,
        ];
        let b_data = vec![
            0., 2., 4., 6., 1., 3., 5., 7., 8., 6., 4., 2., 9., 7., 5., 3., 2., 3., 5., 7., 9., 4.,
            1., 0.,
        ];
        let y_data = vec![
            6420., 246., 6420., 7531., 1357., 7531., 2468., 8642., 2468., 3579., 9753., 3579.,
            7532., 2357., 7532., 149., 9410., 149.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![3, 4], &a_data);
        let b = dev.new_tensor_by_slice(shape![4, 6], &b_data);
        let mut y = dev.new_tensor(shape![3, 6]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_batch_broadcast_1n() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ];
        let b_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
        ];
        let y_data = vec![
            175., 190., 205., 220., 400., 440., 480., 520., 625., 690., 755., 820., 850., 940.,
            1030., 1120., 1075., 1190., 1305., 1420., 1300., 1440., 1580., 1720.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![4, 5], &a_data);
        let b = dev.new_tensor_by_slice(shape![5, 3; 2], &b_data);
        let mut y = dev.new_tensor(shape![4, 3; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_batch_broadcast_n1() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
        ];
        let b_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ];
        let y_data = vec![
            135., 150., 165., 310., 350., 390., 485., 550., 615., 660., 750., 840., 360., 375.,
            390., 910., 950., 990., 1460., 1525., 1590., 2010., 2100., 2190.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![3, 5; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![5, 4], &b_data);
        let mut y = dev.new_tensor(shape![3, 4; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_large() {
        let n = 123;
        let mut a_data = vec![0.; n * n];
        let mut b_data = vec![0.; n * n];
        let mut y1_data = vec![0.; n * n];
        let mut y2_data = vec![0.; n * n];
        let mut k = 0;
        for i in 0..n {
            k += i * i;
        }
        for i in 0..n {
            for j in 0..n {
                a_data[i + j * n] = i as f32 / 16.;
                b_data[i + j * n] = j as f32 / 16.;
                y1_data[i + j * n] = (n * i * j) as f32 / 256.;
                y2_data[i + j * n] = k as f32 / 256.;
            }
        }
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![n as u32, n as u32], &a_data);
        let b = dev.new_tensor_by_slice(shape![n as u32, n as u32], &b_data);
        let mut y1 = dev.new_tensor(shape![n as u32, n as u32]);
        y1.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![n as u32, n as u32]);
        y2.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_matmul_bw_11() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 0., 0., 2.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 5., -3.];
        let gb_data = vec![0., 0., -1., -1.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_ab() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 0., 0., 2.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 5., -3.];
        let gb_data = vec![0., 0., -1., -1.];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_1n() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ];
        let b_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
        ];
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let ga_data = vec![
            1242., 1323., 1404., 1485., 1308., 1395., 1482., 1569., 1374., 1467., 1560., 1653.,
            1440., 1539., 1638., 1737., 1506., 1611., 1716., 1821.,
        ];
        let gb_data = vec![
            31., 71., 111., 151., 191., 71., 175., 279., 383., 487., 111., 279., 447., 615., 783.,
            151., 383., 615., 847., 1079., 191., 487., 783., 1079., 1375., 231., 591., 951., 1311.,
            1671.,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![4, 5], &a_data);
        let b = dev.new_tensor_by_slice(shape![5, 3; 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![4, 3; 2], &gy_data);
        let mut y = dev.new_tensor(shape![4, 3; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![4, 5], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![5, 3; 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_n1() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
        ];
        let b_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ];
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let ga_data = vec![
            263., 297., 331., 285., 323., 361., 307., 349., 391., 329., 375., 421., 351., 401.,
            451., 671., 705., 739., 741., 779., 817., 811., 853., 895., 881., 927., 973., 951.,
            1001., 1051.0,
        ];
        let gb_data = vec![
            731., 875., 1019., 1163., 1307., 902., 1100., 1298., 1496., 1694., 1073., 1325., 1577.,
            1829., 2081., 1244., 1550., 1856., 2162., 2468.0,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![3, 5; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![5, 4], &b_data);
        let gy = dev.new_tensor_by_slice(shape![3, 4; 2], &gy_data);
        let mut y = dev.new_tensor(shape![3, 4; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![3, 5; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![5, 4], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_nn() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
        ];
        let b_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
            37., 38., 39., 40.,
        ];
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let ga_data = vec![
            263., 297., 331., 285., 323., 361., 307., 349., 391., 329., 375., 421., 351., 401.,
            451., 2071., 2185., 2299., 2141., 2259., 2377., 2211., 2333., 2455., 2281., 2407.,
            2533., 2351., 2481., 2611.0,
        ];
        let gb_data = vec![
            15., 33., 51., 69., 87., 33., 78., 123., 168., 213., 51., 123., 195., 267., 339., 69.,
            168., 267., 366., 465., 717., 843., 969., 1095., 1221., 870., 1023., 1176., 1329.,
            1482., 1023., 1203., 1383., 1563., 1743., 1176., 1383., 1590., 1797., 2004.0,
        ];
        let dev = get_device();
        let a = dev.new_tensor_by_slice(shape![3, 5; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![5, 4; 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![3, 4; 2], &gy_data);
        let mut y = dev.new_tensor(shape![3, 4; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![3, 5; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![5, 4; 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
