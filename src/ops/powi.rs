use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(PowiFwImpl, powi_fw_kernel);
impl FunctionFwImpl for PowiFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let k = u32data[0] as i32;
        let y = &mut ys[0];
        let size = y.shape().size();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&k)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(buffer!(y))).unwrap();
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
    }
}

define_opencl_impl_struct!(PowiBwImpl, powi_bw_kernel);
impl FunctionBwImpl for PowiBwImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let x = xs[0];
        let y = ys[0];
        let gy = gys[0];
        let k = u32data[0] as i32;
        let size = y.shape().size();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(gy))).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&k)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 5, ArgVal::mem(buffer!(gx))).unwrap();
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
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_powi_fw() {
        let ns = vec![-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8];
        let x_data = vec![0.1, 2., 1., 2.5, -0.1, 2., -1., -2.5];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        for &n in &ns {
            let y_data = x_data.iter().map(|&x| x.powi(n)).collect::<Vec<f32>>();
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powi_fw_impl", &[&x], &[n as u32], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powi_bw() {
        let ns = vec![-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8];
        let x_data = vec![0.1, 2., 1., 2.5, -0.1, 2., -1., -2.5];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &n in &ns {
            let y_data = x_data.iter().map(|&x| x.powi(n)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * n as f32 * x.powi(n - 1))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powi_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[n as u32],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }
}
