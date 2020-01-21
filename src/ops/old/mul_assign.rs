use std::sync::Mutex;

use ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize;
use ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize as CompileWorkGroupSizeResult;
use ocl::Buffer;
use ocl::Kernel;
use ocl::Program;
use ocl::Queue;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

pub struct MulAssignConstImpl {
    kernel: Mutex<Kernel>,
    wgs_x: u32,
}

impl MulAssignConstImpl {
    pub fn new(program: &Program, queue: Queue) -> MulAssignConstImpl {
        let device = queue.device();
        let kernel = Kernel::builder()
            .program(program)
            .name("mul_assign_const_kernel")
            .queue(queue)
            .arg(0f32)
            .arg(0)
            .arg(None::<&Buffer<f32>>)
            .build()
            .unwrap();
        match kernel.wg_info(device, CompileWorkGroupSize).unwrap() {
            CompileWorkGroupSizeResult([wgs_x, _, _]) => MulAssignConstImpl {
                kernel: Mutex::new(kernel),
                wgs_x: wgs_x as u32,
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for MulAssignConstImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        let size = y.shape().size();
        let g1 = super::calc_num_blocks(size, self.wgs_x);
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            kernel.set_arg(0, k).unwrap();
            kernel.set_arg(1, size).unwrap();
            kernel.set_arg(2, buffer!(y)).unwrap();
            kernel
                .cmd()
                .global_work_size([g1 * self.wgs_x, 1, 1])
                .local_work_size([self.wgs_x, 1, 1])
                .enq()
                .unwrap();
        }
    }
}
