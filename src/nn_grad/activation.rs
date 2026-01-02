use crate::linalg::tensor_grad::Tensor;
use crate::nn_grad::{Dumpable, Layer};
use std::fs::File;
use std::io::BufReader;

#[derive(Default)]
pub struct ReLU {}

impl Dumpable for ReLU {
    fn restore(_reader: &mut BufReader<File>) -> Box<dyn Dumpable>
    where
        Self: Sized,
    {
        Box::new(ReLU {})
    }
    fn type_id() -> &'static str {
        "relu"
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
}

#[derive(Default)]
pub struct LogSoftmax;

impl Dumpable for LogSoftmax {
    fn restore(_reader: &mut BufReader<File>) -> Box<dyn Dumpable>
    where
        Self: Sized,
    {
        Box::new(LogSoftmax {})
    }
    fn type_id() -> &'static str {
        "log_softmax"
    }
}

impl Layer for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.log_softmax()
    }
}

#[derive(Default)]
pub struct Softmax;

impl Dumpable for Softmax {
    fn restore(_reader: &mut BufReader<File>) -> Box<dyn Dumpable>
    where
        Self: Sized,
    {
        Box::new(Softmax {})
    }
    fn type_id() -> &'static str {
        "softmax"
    }
}

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax()
    }
}

#[derive(Default)]
pub struct Sigmoid;
impl Dumpable for Sigmoid {
    fn restore(_reader: &mut BufReader<File>) -> Box<dyn Dumpable>
    where
        Self: Sized,
    {
        Box::new(Sigmoid {})
    }
    fn type_id() -> &'static str {
        "sigmoid"
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }
}
