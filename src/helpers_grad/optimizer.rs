use crate::linalg::tensor_grad::{Scalar, Tensor};

pub trait Optimizer {
    fn step(&mut self, params: Vec<&mut Tensor>);
    fn reset(&mut self) {}
}

pub struct SGD {
    learning_rate: Scalar,
}

impl SGD {
    pub fn new(learning_rate: Scalar) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: Vec<&mut Tensor>) {
        for param in params {
            if !param.requires_grad {
                continue;
            }
            let grad = param.grad().expect("Gradient not found for parameter");
            *param = &*param - &(grad * self.learning_rate);
            param.clear_graph();
        }
    }
}

pub struct Adam {
    learning_rate: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    mean_vectors: Vec<Tensor>,
    variance_vectors: Vec<Tensor>,
    time_step: usize,
}

impl Adam {
    pub fn new(learning_rate: Scalar, beta1: Scalar, beta2: Scalar, epsilon: Scalar) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            mean_vectors: Vec::new(),
            variance_vectors: Vec::new(),
            time_step: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: Vec<&mut Tensor>) {
        let mut i = 0;
        for param in params {
            if !param.requires_grad {
                continue;
            }
            let grad = param.grad().expect("Gradient not found for parameter");
            if i >= self.mean_vectors.len() {
                self.mean_vectors.push(Tensor::new(
                    vec![0.0; param.storage.data.len()],
                    param.shape(),
                ));
                self.variance_vectors.push(Tensor::new(
                    vec![0.0; param.storage.data.len()],
                    param.shape(),
                ));
            }

            let m = &self.mean_vectors[i] * self.beta1 + &grad * (1.0 - self.beta1);
            let v = &self.variance_vectors[i] * self.beta2 + grad.square() * (1.0 - self.beta2);

            let m_hat = &m / (1.0 - self.beta1.powi((self.time_step + 1) as i32));
            let v_hat = &v / (1.0 - self.beta2.powi((self.time_step + 1) as i32));

            *param = &*param - &(&m_hat / &(&v_hat.sqrt() + self.epsilon) * self.learning_rate);
            self.mean_vectors[i] = m;
            self.variance_vectors[i] = v;
            self.time_step += 1;
            i += 1;
        }
    }

    fn reset(&mut self) {
        self.mean_vectors.clear();
        self.variance_vectors.clear();
        self.time_step = 0;
    }
}
