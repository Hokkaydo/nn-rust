use crate::linalg::tensor_old::Tensor;
use crate::nn::activation::{LogSoftmax, ReLU, Sigmoid};
use crate::nn::linear::Linear;
use crate::nn::memory::Memory;
use crate::nn::{Dumpable, Layer};
use std::io::{BufRead, Write};

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
    memory: Memory,
}

impl NeuralNetwork {
    pub fn init(layers: Vec<Box<dyn Layer>>, memory: Memory) -> Self {
        NeuralNetwork { layers, memory }
    }

    pub fn new() -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            memory: Memory::new(),
        }
    }

    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let mut output = input;
        for layer in &mut self.layers {
            output = layer.forward(&mut self.memory, &output);
        }
        output
    }

    pub fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let mut grad = grad_output;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&mut self.memory, &grad);
        }
        grad
    }

    pub fn apply_gradients(&mut self, update_function: &dyn Fn(Vec<&Tensor>) -> Vec<Tensor>) {
        for layer in &mut self.layers {
            layer.apply_gradients(&mut self.memory, update_function);
        }
    }

    pub fn dump_memory(&self, path: &str) {
        let file = std::fs::File::create(format!("{}", path)).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for layer in &self.layers {
            writer
                .write_all(format!("{}\n", layer.type_id()).as_bytes())
                .unwrap();
            layer.dump(&self.memory, &mut writer);
        }
    }

    pub fn restore_memory(&mut self, path: &str) {
        let file = std::fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(file);
        let mut token = String::new();
        while reader.read_line(&mut token).unwrap() > 0 {
            token.pop();
            let mut layer: Box<dyn Layer> = match token.as_str() {
                "linear" => Box::new(Linear::new()),
                "relu" => Box::new(ReLU::new()),
                "log_softmax" => Box::new(LogSoftmax::new()),
                "sigmoid" => Box::new(Sigmoid::new()),
                _ => panic!("Unknown layer type: {}", token),
            };
            layer.restore(&mut self.memory, &mut reader);
            self.layers.push(layer);
            token.clear();
        }
    }
}
