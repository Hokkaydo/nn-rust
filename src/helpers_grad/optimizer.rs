use crate::linalg::tensor_grad::{Scalar, Tensor};

pub fn gradient_descent(learning_rate: Scalar) -> Box<dyn Fn(Vec<&Tensor>) -> Vec<Tensor>> {
    Box::new(move |values: Vec<&Tensor>| {
        let val = values[0];
        let grad = values[1];
        let out = val - &(grad * learning_rate);
        vec![out]
    })
}

pub fn adam(
    learning_rate: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
) -> Box<dyn Fn(Vec<&Tensor>) -> Vec<Tensor>> {
    Box::new(move |values: Vec<&Tensor>| {
        let val = values[0];
        let grad = values[1];

        let m = if values.len() > 2 {
            values[2].clone()
        } else {
            Tensor::new(vec![0.0; val.storage.data.len()], val.shape())
        };
        let v = if values.len() > 3 {
            values[3].clone()
        } else {
            Tensor::new(vec![0.0; val.storage.data.len()], val.shape())
        };

        let t = if values.len() > 4 {
            values[4]
        } else {
            &Tensor::new(vec![1.0], &[1])
        };

        let m = &m * beta1 + grad * (1.0 - beta1);
        let v = &v * beta2 + grad.square() * (1.0 - beta2);

        let m_hat = &m / (1.0 - beta1.powi(t.storage.data[0] as i32));
        let v_hat = &v / (1.0 - beta2.powi(t.storage.data[0] as i32));

        let out = val - &(&m_hat / &(&v_hat.sqrt() + epsilon) * learning_rate);
        vec![out, m, v, t + 1.0]
    })
}
