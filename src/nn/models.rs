use crate::linalg::tensor::Tensor;
use crate::nn::{Layer, registry};
use std::any::Any;
use std::io::{BufRead, Write};
use std::ops::Deref;

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn init(layers: Vec<Box<dyn Layer>>) -> Self {
        NeuralNetwork { layers }
    }

    pub fn restore(path: &str) -> Self {
        let mut nn = NeuralNetwork { layers: vec![] };
        nn.restore_memory(path);
        nn
    }

    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let mut output = input;
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    pub fn dump_memory(&self, path: &str) {
        let file = std::fs::File::create(path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for layer in &self.layers {
            writer
                .write_all(format!("{:?}\n", layer.deref().type_id()).as_bytes())
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
                .unwrap_or_else(|| panic!("Unknown type_id: {type_id}"));
            restore_fn(&mut reader);
            token.clear();
        }
    }
}
