use std::ops::Mul;
use std::ops::*;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data length does not match shape dimensions"
        );
        Tensor { data, shape }
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let flat_index = self.flatten_index(indices);
        self.data[flat_index]
    }

    pub fn get_level(&self, index: usize) -> Tensor {
        assert!(
            index < self.shape[0],
            "Index out of bounds for tensor dimensions"
        );
        let mut new_shape = self.shape.clone();
        new_shape[0] = 1;
        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        new_data.extend_from_slice(
            &self.data[index * new_shape.iter().product::<usize>()
                ..(index + 1) * new_shape.iter().product::<usize>()],
        );

        Tensor::new(new_data, new_shape)
    }

    pub fn get_levels(&self, indices: &[usize]) -> Tensor {
        assert!(
            indices.len() != 0
                && indices
                    .iter()
                    .map(|&x| x >= self.shape[0])
                    .filter(|&x| x)
                    .count()
                    == 0,
            "Indices length exceeds tensor dimensions"
        );
        let mut results = Vec::new();
        for &index in indices {
            results.extend_from_slice(&self.get_level(index).data);
        }
        let mut new_shape = self.shape.clone();
        new_shape[0] = indices.len();
        Tensor::new(results, new_shape)
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let flat_index = self.flatten_index(indices);
        self.data[flat_index] = value;
    }
    pub fn square(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x * x).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn abs(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.abs()).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn log(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.ln()).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn mean(&self) -> f32 {
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
    }

    pub fn argmax(&self) -> usize {
        assert!(!self.data.is_empty(), "Tensor is empty");
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(i, _)| i)
    }

    pub fn map<F>(&self, func: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.data.iter().map(|&x| func(x)).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn reduce<F>(&self, default: f32, func: F) -> f32
    where
        F: Fn(f32, f32) -> f32,
    {
        if self.data.is_empty() {
            return default;
        }
        self.data.iter().fold(default, |acc, &x| func(acc, x))
    }

    pub fn max(&self) -> f32 {
        self.reduce(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(
            self.shape.len(),
            2,
            "Transpose is only defined for 1D and 2D tensors"
        );
        let mut new_data = Vec::with_capacity(self.data.len());
        for j in 0..self.shape[1] {
            for i in 0..self.shape[0] {
                new_data.push(self.data[i * self.shape[1] + j]);
            }
        }
        Tensor::new(new_data, vec![self.shape[1], self.shape[0]])
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(), "Axis out of bounds");

        let mut new_shape = self.shape.clone();
        let axis_dim = new_shape[axis];
        new_shape[axis] = 1;

        let total_elems: usize = self.shape.iter().product();
        let mut result_data = vec![0.0; total_elems / axis_dim];

        let outer = self.shape[..axis].iter().product::<usize>();
        let inner = self.shape[(axis + 1)..].iter().product::<usize>();

        for o in 0..outer {
            for i in 0..inner {
                let mut sum = 0.0;
                for a in 0..axis_dim {
                    let idx = o * axis_dim * inner + a * inner + i;
                    sum += self.data[idx];
                }
                let result_idx = o * inner + i;
                result_data[result_idx] = sum;
            }
        }

        Tensor {
            data: result_data,
            shape: new_shape,
        }
    }

    pub fn flatten_index(&self, indices: &[usize]) -> usize {
        let shape = if indices.len() == self.shape.len() {
            &self.shape
        } else {
            &self
                .shape
                .iter()
                .filter(|&i| *i > 1)
                .map(|&i| i)
                .collect::<Vec<_>>()
        };
        if indices.len() != shape.len() {
            panic!(
                "Indices length {} does not match tensor shape length {}",
                indices.len(),
                shape.len()
            );
        }
        let strides = Tensor::compute_strides(shape);
        indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }

    pub fn squeeze(&self) -> Tensor {
        let new_shape: Vec<usize> = self.shape.iter().filter(|&&dim| dim > 1).cloned().collect();
        if new_shape.is_empty() {
            return Tensor::new(vec![0.0], vec![1]);
        }
        let new_data: Vec<f32> = self.data.iter().cloned().collect();
        Tensor::new(new_data, new_shape)
    }

    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape.len(),
            other.shape.len(),
            "Tensors must have same rank"
        );

        let output_shape = self
            .shape
            .iter()
            .zip(&other.shape)
            .map(|(&a, &b)| {
                if a == b {
                    a
                } else if a == 1 {
                    b
                } else if b == 1 {
                    a
                } else {
                    panic!(
                        "Incompatible shapes for broadcasting: {:?} and {:?}",
                        self.shape, other.shape
                    );
                }
            })
            .collect::<Vec<_>>();

        let output_size = output_shape.iter().product::<usize>();
        let mut output_data = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let idx = Tensor::unravel_index_generic(i, &output_shape);

            let self_idx = Tensor::flatten_index_broadcast(&idx, &self.shape);
            let other_idx = Tensor::flatten_index_broadcast(&idx, &other.shape);

            output_data.push(self.data[self_idx] + other.data[other_idx]);
        }

        Tensor {
            data: output_data,
            shape: output_shape,
        }
    }

    fn flatten_index_broadcast(index: &[usize], shape: &[usize]) -> usize {
        let strides = Tensor::compute_strides(shape);
        index
            .iter()
            .zip(shape.iter())
            .zip(strides.iter())
            .map(|((&i, &dim), &stride)| {
                let real_i = if dim == 1 { 0 } else { i };
                real_i * stride
            })
            .sum()
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn unravel_index_generic(mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            coords[i] = index % shape[i];
            index /= shape[i];
        }
        coords
    }

    pub fn element_wise_multiply(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape for element-wise multiplication"
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let flat_index = self.flatten_index(indices);
        &self.data[flat_index]
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let flat_index = self.flatten_index(indices);
        &mut self.data[flat_index]
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Tensor {
        &self + &other
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape for addition"
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        self + &other
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        &self + other
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Tensor {
        &self + scalar
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| self + x).collect();
        Tensor::new(data, tensor.shape.clone())
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Tensor {
        &tensor + self
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        -&self
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape for subtraction"
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        self - &other
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        &self - other
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        &self - &other
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x - scalar).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Tensor {
        &self - scalar
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: Tensor) -> Tensor {
        self - &tensor
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| self - x).collect();
        Tensor::new(data, tensor.shape.clone())
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Tensor {
        &self * &tensor
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Tensor {
        self * &tensor
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Tensor {
        &self * tensor
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, tensor: &Tensor) -> Tensor {
        if self.shape.len() == 1 && tensor.shape.len() == 1 {
            // Element-wise multiplication for 1D tensors
            assert_eq!(
                self.shape, tensor.shape,
                "1D tensors must have the same shape for multiplication"
            );
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(&tensor.data)
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(data, self.shape.clone());
        }

        if self.shape.len() == 2 && tensor.shape.len() == 2 {
            // Matrix multiplication for 2D tensors
            assert_eq!(
                self.shape[1], tensor.shape[0],
                "Inner dimensions must match for matrix multiplication"
            );
            let mut data = vec![0.0; self.shape[0] * tensor.shape[1]];
            for i in 0..self.shape[0] {
                for j in 0..tensor.shape[1] {
                    for k in 0..self.shape[1] {
                        data[i * tensor.shape[1] + j] +=
                            self.data[i * self.shape[1] + k] * tensor.data[k * tensor.shape[1] + j];
                    }
                }
            }
            return Tensor::new(data, vec![self.shape[0], tensor.shape[1]]);
        }

        if self.shape.len() == 2 && tensor.shape.len() == 1 {
            // Matrix-vector multiplication
            assert_eq!(
                self.shape[1], tensor.shape[0],
                "Inner dimensions must match for matrix-vector multiplication"
            );
            let mut data = vec![0.0; self.shape[0]];
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    data[i] += self.data[i * self.shape[1] + j] * tensor.data[j];
                }
            }
            return Tensor::new(data, vec![self.shape[0]]);
        }

        if self.shape.len() == 1 && tensor.shape.len() == 2 {
            // Vector-matrix multiplication
            assert_eq!(
                self.shape[0], tensor.shape[0],
                "Inner dimensions must match for vector-matrix multiplication"
            );
            let mut data = vec![0.0; tensor.shape[1]];
            for i in 0..tensor.shape[1] {
                for j in 0..self.shape[0] {
                    data[i] += self.data[j] * tensor.data[j * tensor.shape[1] + i];
                }
            }
            return Tensor::new(data, vec![tensor.shape[1]]);
        }
        panic!(
            "Multiplication is only defined for 1D or 2D tensors. Got shapes {:?} and {:?}",
            self.shape, tensor.shape
        );
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Tensor {
        &tensor * self
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| x * self).collect();
        Tensor::new(data, tensor.shape.clone())
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        &self * scalar
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Tensor {
        if scalar == 0.0 {
            panic!("Division by zero is not allowed");
        }
        let data: Vec<f32> = self.data.iter().map(|&x| x / scalar).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Tensor {
        &self / scalar
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, tensor: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, tensor.shape,
            "Tensors must have the same shape for division"
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&tensor.data)
            .map(|(a, b)| {
                if *b == 0.0 {
                    panic!("Division by zero is not allowed");
                }
                a / b
            })
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        &self / &tensor
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        self / &tensor
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, tensor: &Tensor) -> Tensor {
        &self / tensor
    }
}
