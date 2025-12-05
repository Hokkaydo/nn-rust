use crate::linalg::tensor::Tensor;
use crate::nn::memory::Memory;
use crate::nn::{Dumpable, Layer};

pub struct ReLU {}

impl Dumpable for ReLU {
    fn new() -> Self {
        ReLU {}
    }
    fn type_id(&self) -> &'static str {
        "relu"
    }
}

impl Layer for ReLU {
    fn forward(&mut self, _memory: &mut Memory, input: &Tensor) -> Tensor {
        input.map(|input| if input < 0.0 { 0.0 } else { input })
    }
    fn backward(&mut self, _memory: &mut Memory, grad_output: &Tensor) -> Tensor {
        grad_output.map(|grad| if grad < 0.0 { 0.0 } else { grad })
    }
}

pub struct LogSoftmax {
    forward_pass_result_index: usize,
}

impl Dumpable for LogSoftmax {
    fn new() -> Self {
        LogSoftmax {
            forward_pass_result_index: 0,
        }
    }
    fn type_id(&self) -> &'static str {
        "log_softmax"
    }
}

impl Layer for LogSoftmax {
    fn forward(&mut self, _memory: &mut Memory, input: &Tensor) -> Tensor {
        let max = input.max();
        let exp_data = input.map(|x| (x - max).exp());
        let sum_exp: f32 = exp_data.reduce(0.0, |acc, x| acc + x);
        let log_softmax = exp_data.map(|x| (x / sum_exp).ln());
        self.forward_pass_result_index = _memory.push(log_softmax.clone());
        log_softmax
    }

    fn backward(&mut self, _memory: &mut Memory, grad_output: &Tensor) -> Tensor {
        let log_softmax = _memory.get(self.forward_pass_result_index);
        let sum_grad = grad_output.reduce(0.0, |acc, x| acc + x);

        let grad_input = grad_output
            .data
            .iter()
            .zip(log_softmax.data.iter())
            .map(|(&grad, &log_soft)| grad - (log_soft * sum_grad))
            .collect::<Vec<f32>>();
        Tensor::new(grad_input, grad_output.shape.clone())
    }
}

pub struct Softmax {}

impl Dumpable for Softmax {
    fn new() -> Self {
        Softmax {}
    }
    fn type_id(&self) -> &'static str {
        "softmax"
    }
}

impl Layer for Softmax {
    fn forward(&mut self, _memory: &mut Memory, input: &Tensor) -> Tensor {
        let max = input.max();
        let exp_data = input.map(|x| (x - max).exp());
        let sum_exp: f32 = exp_data.reduce(1e-6, |acc, x| acc + x);
        exp_data.map(|x| x / sum_exp)
    }

    fn backward(&mut self, _memory: &mut Memory, grad_output: &Tensor) -> Tensor {
        let grad_input = grad_output.map(|grad| {
            let exp_grad = grad.exp();
            exp_grad * (1.0 - exp_grad)
        });
        grad_input
    }
}

pub struct Sigmoid {
    sigmoid_index: usize,
}
impl Dumpable for Sigmoid {
    fn new() -> Self {
        Sigmoid { sigmoid_index: 0 }
    }
    fn type_id(&self) -> &'static str {
        "sigmoid"
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, _memory: &mut Memory, input: &Tensor) -> Tensor {
        let sigmoid = input.map(|x| 1.0 / (1.0 + (-x).exp()));
        self.sigmoid_index = _memory.push(sigmoid.clone());
        sigmoid
    }

    fn backward(&mut self, mem: &mut Memory, grad_output: &Tensor) -> Tensor {
        let sigmoid = mem.get(self.sigmoid_index);
        sigmoid
            .element_wise_multiply(&(1.0 - sigmoid))
            .element_wise_multiply(grad_output)
    }
}
