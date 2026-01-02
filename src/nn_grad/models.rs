use crate::linalg::tensor_grad::Tensor;
use crate::nn_grad::{Dumpable, Layer, registry};
use std::any::Any;
use std::io::{BufRead, Write};

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn init(layers: Vec<Box<dyn Layer>>) -> Self {
        NeuralNetwork { layers }
    }

    pub fn restore(path: &str) -> Self {
        let mut nn = NeuralNetwork { layers: vec![] };
        nn.restore_memory(&path);
        nn
    }

    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let mut output = input;
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let mut grad = grad_output;
        for layer in self.layers.iter_mut().rev() {
            // grad = layer.backward(&mut self.memory, &grad);
        }
        grad
    }

    pub fn dump_memory(&self, path: &str) {
        let file = std::fs::File::create(format!("{}", path)).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for layer in &self.layers {
            writer
                .write_all(format!("{:?}\n", layer.type_id()).as_bytes())
                .unwrap();
            layer.dump(&mut writer);
        }
    }

    fn restore_memory(&mut self, path: &str) {
        let file = std::fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(file);
        let mut token = String::new();
        while reader.read_line(&mut token).unwrap() > 0 {
            token.pop();
            let type_id = token.as_str();
            let map = registry();
            let restore_fn = map
                .get(type_id)
                .unwrap_or_else(|| panic!("Unknown type_id: {}", type_id));
            restore_fn(&mut reader);
            token.clear();
        }
    }
}
