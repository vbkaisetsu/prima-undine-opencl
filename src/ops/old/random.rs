use std::sync::Arc;
use std::sync::Mutex;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Randomizer;
use prima_undine::Tensor;

pub struct RandomBernoulliImpl {
    randomizer: Arc<Mutex<Box<dyn Randomizer>>>,
}

impl RandomBernoulliImpl {
    pub fn new(randomizer: Arc<Mutex<Box<dyn Randomizer>>>) -> RandomBernoulliImpl {
        RandomBernoulliImpl {
            randomizer: randomizer,
        }
    }
}

impl FunctionFwImpl for RandomBernoulliImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let p = f32data[0];
        let y = &mut ys[0];
        unsafe {
            let mut tmp = vec![0.; y.shape().size() as usize];
            self.randomizer.lock().unwrap().fill_bernoulli(p, &mut tmp);
            buffer!(y).write(&tmp).enq().unwrap();
        }
    }
}

pub struct RandomNormalImpl {
    randomizer: Arc<Mutex<Box<dyn Randomizer>>>,
}

impl RandomNormalImpl {
    pub fn new(randomizer: Arc<Mutex<Box<dyn Randomizer>>>) -> RandomNormalImpl {
        RandomNormalImpl {
            randomizer: randomizer,
        }
    }
}

impl FunctionFwImpl for RandomNormalImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let mean = f32data[0];
        let sd = f32data[1];
        let y = &mut ys[0];
        unsafe {
            let mut tmp = vec![0.; y.shape().size() as usize];
            self.randomizer
                .lock()
                .unwrap()
                .fill_normal(mean, sd, &mut tmp);
            buffer!(y).write(&tmp).enq().unwrap();
        }
    }
}
