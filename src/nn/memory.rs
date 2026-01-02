use crate::linalg::tensor_old::Tensor;

pub struct Memory {
    size: usize,
    tensors: Vec<Tensor>,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            size: 0,
            tensors: Vec::new(),
        }
    }

    pub fn push(&mut self, tensor: Tensor) -> usize {
        self.tensors.push(tensor);
        self.size += 1;
        self.size - 1
    }

    pub fn new_push(&mut self, shape: &[usize]) -> usize {
        let data_size = shape.iter().product();
        let tensor = Tensor::new(vec![0.0; data_size], shape.to_vec());
        self.push(tensor)
    }

    pub fn get(&self, index: usize) -> &Tensor {
        assert!(
            index < self.tensors.len(),
            "Index out of bounds for memory tensors"
        );
        &self.tensors[index]
    }

    pub fn alter<F>(&mut self, index: usize, f: F)
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        assert!(
            index < self.tensors.len(),
            "Index out of bounds for memory tensors"
        );
        let current = self.tensors[index].clone();
        self.tensors[index] = f(&current);
    }

    pub fn set(&mut self, index: usize, tensor: Tensor) {
        assert!(
            index < self.tensors.len(),
            "Index out of bounds for memory tensors"
        );
        self.tensors[index] = tensor;
    }
}
