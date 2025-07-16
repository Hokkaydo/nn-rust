use crate::linalg::tensor::Tensor;
use crate::nn::memory::Memory;
use crate::nn::{Dumpable, Layer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};

struct Conv2D {
    params: HashMap<String, usize>,
}

impl Conv2D {
    pub fn init(
        mem: &mut Memory,
        kernel_size: usize,
        in_channel: usize,
        out_channel: usize,
    ) -> Self {
        let mut params: HashMap<String, usize> = HashMap::new();

        params.insert(
            "kernel".to_string(),
            mem.new_push(&[out_channel, in_channel, kernel_size, kernel_size]),
        );
        params.insert(
            "bias".to_string(),
            mem.push(Tensor::full(vec![out_channel], 0.0)),
        );

        Conv2D { params }
    }
}

impl Dumpable for Conv2D {
    fn dump(&self, mem: &Memory, file: &mut std::io::BufWriter<std::fs::File>) {
        let kernel_index = self.params.get("kernel").unwrap();
        let bias_index = self.params.get("bias").unwrap();
        let kernel = mem.get(*kernel_index);
        let bias = mem.get(*bias_index);

        let mut sizes = kernel
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

        file.write_all(&sizes).unwrap();
        file.write_all(self.f32_to_bytes(&kernel.data).as_slice())
            .unwrap();
        file.write_all(self.f32_to_bytes(&bias.data).as_slice())
            .unwrap();
    }

    fn restore(&mut self, mem: &mut Memory, file: &mut BufReader<File>) {
        let mut sizes = vec![0u8; 4 * 4]; // Assuming 4 dimensions for kernel and bias
        file.read_exact(&mut sizes).unwrap();

        let kernel_shape = [
            u32::from_le_bytes([sizes[0], sizes[1], sizes[2], sizes[3]]) as usize,
            u32::from_le_bytes([sizes[4], sizes[5], sizes[6], sizes[7]]) as usize,
            u32::from_le_bytes([sizes[8], sizes[9], sizes[10], sizes[11]]) as usize,
            u32::from_le_bytes([sizes[12], sizes[13], sizes[14], sizes[15]]) as usize,
        ];

        let bias_shape =
            [u32::from_le_bytes([sizes[16], sizes[17], sizes[18], sizes[19]]) as usize];

        let kernel_size = kernel_shape.iter().product::<usize>();
        let bias_size = bias_shape.iter().product::<usize>();
        let mut kernel_data = vec![0.0; kernel_size];
        let mut bias_data = vec![0.0; bias_size];

        file.read_exact(bytemuck::cast_slice_mut(&mut kernel_data))
            .unwrap();
        file.read_exact(bytemuck::cast_slice_mut(&mut bias_data))
            .unwrap();

        let kernel_tensor = Tensor::new(kernel_data, kernel_shape.to_vec());
        let bias_tensor = Tensor::new(bias_data, bias_shape.to_vec());

        self.params
            .insert("kernel".to_string(), mem.push(kernel_tensor));
        self.params
            .insert("bias".to_string(), mem.push(bias_tensor));
    }

    fn type_id(&self) -> &'static str {
        "conv2d"
    }
}

impl Layer for Conv2D {
    fn new() -> Self
    where
        Self: Sized,
    {
        Conv2D {
            params: HashMap::new(),
        }
    }

    fn forward(&mut self, mem: &mut Memory, input: &Tensor) -> Tensor {
        let kernel_index = self.params.get("kernel").unwrap();
        let bias_index = self.params.get("bias").unwrap();
        let kernel = mem.get(*kernel_index);
        let bias = mem.get(*bias_index);

        // Implement the convolution operation here
        // This is a placeholder implementation
        let output_shape = vec![
            input.shape[0],                             // Batch size
            kernel.shape[0],                            // Output channels
            (input.shape[2] - kernel.shape[2]) / 1 + 1, // Height
            (input.shape[3] - kernel.shape[3]) / 1 + 1, // Width
        ];

        let output_data = vec![0.0; output_shape.iter().product()]; // Placeholder for actual convolution result

        Tensor::new(output_data, output_shape)
    }

    fn backward(&mut self, mem: &mut Memory, grad_output: &Tensor) -> Tensor {
        todo!()
    }

    fn apply_gradients(
        &mut self,
        _mem: &mut Memory,
        _update_function: &dyn Fn(Vec<&Tensor>) -> Vec<Tensor>,
    ) {
        todo!()
    }
}
