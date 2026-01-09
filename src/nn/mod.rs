pub mod activation;
pub mod linear;
pub mod models;

use crate::linalg::tensor::Tensor;
use crate::nn::activation::{LogSoftmax, ReLU, Softmax};
use crate::nn::linear::Linear;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::OnceLock;

type RestoreFn = fn(&mut BufReader<File>) -> Box<dyn Dumpable>;

static REGISTRY: OnceLock<HashMap<&'static str, RestoreFn>> = OnceLock::new();
fn registry() -> &'static HashMap<&'static str, RestoreFn> {
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();

        m.insert(Linear::type_id(), Linear::restore as RestoreFn);
        m.insert(ReLU::type_id(), ReLU::restore as RestoreFn);
        m.insert(LogSoftmax::type_id(), LogSoftmax::restore as RestoreFn);
        m.insert(Softmax::type_id(), Softmax::restore as RestoreFn);

        m
    })
}

pub trait Layer: Dumpable {
    /// Forward function takes an input tensor and returns the output tensor after applying the layer's operation.
    /// # Arguments
    /// * `input` - A reference to the input Tensor.
    /// # Returns
    /// * A Tensor representing the output after applying the layer's operation.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns a vector of references to the layer's parameters (weights, biases, etc.).
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    /// Returns a vector of mutable references to the layer's parameters (weights, biases, etc.).
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

pub trait Dumpable {
    fn dump(&self, _file: &mut BufWriter<File>) {}
    fn restore(_reader: &mut BufReader<File>) -> Box<dyn Dumpable>
    where
        Self: Sized;
    fn type_id() -> &'static str
    where
        Self: Sized;
    fn type_id_instance(&self) -> &'static str
    where
        Self: Sized,
    {
        Self::type_id()
    }
}
