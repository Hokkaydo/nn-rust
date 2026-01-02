use crate::linalg::tensor_old::Tensor;
use crate::nn::memory::Memory;
use crate::nn::{Dumpable, Layer};
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub struct Linear {
    params: HashMap<String, usize>,
}

impl Linear {
    pub fn init(mem: &mut Memory, n_inputs: usize, n_outputs: usize) -> Self {
        let mut params: HashMap<String, usize> = HashMap::new();
        // let fan_avg = (n_inputs + n_outputs) as f32 / 2.0;
        // let range = rand::distr::Uniform::new(-6.0 / fan_avg.sqrt(), 6.0 / fan_avg.sqrt()).unwrap();
        let range = rand::distr::Uniform::new(
            -6.0 / (n_inputs as f32).sqrt(),
            6.0 / (n_inputs as f32).sqrt(),
        )
        .unwrap();
        let weights: Vec<f32> = rand::rng()
            .sample_iter(range)
            .take(n_inputs * n_outputs)
            .collect();
        let bias: Vec<f32> = rand::rng().sample_iter(range).take(n_outputs).collect();

        params.insert(
            "weights".to_string(),
            mem.push(Tensor::new(weights, vec![n_inputs, n_outputs])),
        );
        params.insert(
            "bias".to_string(),
            mem.push(Tensor::new(bias, vec![1, n_outputs])),
        );
        Linear { params }
    }
}

impl Dumpable for Linear {
    fn new() -> Self {
        Linear {
            params: HashMap::new(),
        }
    }
    fn dump(&self, mem: &Memory, file: &mut BufWriter<File>) {
        let weights_index = self.params.get("weights").unwrap();
        let bias_index = self.params.get("bias").unwrap();
        let weights = mem.get(*weights_index);
        let bias = mem.get(*bias_index);

        let mut sizes = weights
            .shape
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect::<Vec<u8>>();
        sizes.append(
            &mut bias
                .shape
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );

        file.write_all(&sizes)
            .expect("Unable to write sizes to file");
        file.write_all(
            &weights
                .data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Unable to write weights to file");
        file.write_all(
            &bias
                .data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Unable to write bias to file");
    }
    fn restore(&mut self, mem: &mut Memory, file: &mut BufReader<File>) {
        let mut sizes = [0u8; 32]; // 4 * 8 bytes for 4 usize values
        file.read_exact(&mut sizes)
            .expect("Unable to read sizes from file");

        let weights_shape = (
            usize::from_le_bytes(sizes[0..8].try_into().unwrap()),
            usize::from_le_bytes(sizes[8..16].try_into().unwrap()),
        );
        let bias_shape = (
            usize::from_le_bytes(sizes[16..24].try_into().unwrap()),
            usize::from_le_bytes(sizes[24..32].try_into().unwrap()),
        );

        let weights_size = weights_shape.0 * weights_shape.1;
        let bias_size = bias_shape.1;

        let mut weights_data = vec![0.0; weights_size];
        let mut bias_data = vec![0.0; bias_size];

        file.read_exact(bytemuck::cast_slice_mut(&mut weights_data))
            .expect("Unable to read weights from file");
        file.read_exact(bytemuck::cast_slice_mut(&mut bias_data))
            .expect("Unable to read bias from file");

        self.params.insert(
            "weights".to_string(),
            mem.push(Tensor::new(
                weights_data,
                vec![weights_shape.0, weights_shape.1],
            )),
        );
        self.params.insert(
            "bias".to_string(),
            mem.push(Tensor::new(bias_data, vec![bias_shape.0, bias_shape.1])),
        );
    }
    fn type_id(&self) -> &'static str {
        "linear"
    }
}

impl Layer for Linear {
    fn forward(&mut self, mem: &mut Memory, input: &Tensor) -> Tensor {
        self.params
            .insert("input".to_string(), mem.push(input.clone()));
        let weights = mem.get(*self.params.get("weights").unwrap());
        let bias = mem.get(*self.params.get("bias").unwrap());

        (input * weights).broadcast_add(bias)
    }

    fn backward(&mut self, mem: &mut Memory, grad_output: &Tensor) -> Tensor {
        let weights_index = *self.params.get("weights").unwrap();
        let weights = mem.get(weights_index);
        let input = mem.get(*self.params.get("input").unwrap());

        let grad_input = grad_output * weights.transpose();
        let grad_weights = input.transpose() * grad_output;
        let grad_bias = grad_output.sum_axis(0);

        fn update_grad_batch(
            params: &mut HashMap<String, usize>,
            name: &str,
            mem: &mut Memory,
            grad: Tensor,
        ) {
            let batch_index = params.get(name);
            if let Some(index) = batch_index {
                mem.alter(*index, |w| w + grad);
            } else {
                let batch_index = mem.push(Tensor::new(grad.data.clone(), grad.shape.clone()));
                params.insert(name.to_string(), batch_index);
            }
        }
        update_grad_batch(&mut self.params, "weights_batch", mem, grad_weights);
        update_grad_batch(&mut self.params, "bias_batch", mem, grad_bias);

        grad_input
    }

    fn apply_gradients(
        &mut self,
        mem: &mut Memory,
        update_function: &dyn Fn(Vec<&Tensor>) -> Vec<Tensor>,
    ) {
        let weights_batch_index = self.params.get("weights_batch").unwrap();
        let bias_batch_index = self.params.get("bias_batch").unwrap();

        let weights_index = self.params.get("weights").unwrap();
        let bias_index = self.params.get("bias").unwrap();

        let weights_batch = mem.get(*weights_batch_index).clone();
        let bias_batch = mem.get(*bias_batch_index).clone();

        let weights = mem.get(*weights_index).clone();
        let bias = mem.get(*bias_index).clone();

        let mut base_update = |param: &Tensor, batch: &Tensor, param_name: &str| {
            self.update_param(
                self.params.clone(),
                mem,
                update_function,
                param,
                batch,
                param_name,
            )
        };

        let weights = base_update(&weights, &weights_batch, "weights");
        let bias = base_update(&bias, &bias_batch, "bias");

        mem.set(*weights_index, weights);
        mem.set(*bias_index, bias);

        mem.alter(*weights_batch_index, |w| {
            Tensor::new(vec![0.0; w.data.len()], w.shape.clone())
        });
        mem.alter(*bias_batch_index, |b| {
            Tensor::new(vec![0.0; b.data.len()], b.shape.clone())
        });
    }
}
