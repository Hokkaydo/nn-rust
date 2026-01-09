use crate::linalg::autograd::grad_fn::reduce::{MeanGradFn, SumAxisGradFn, SumGradFn};
use crate::linalg::tensor::{InternalTensor, Scalar, Storage, Tensor};
use crate::not_implemented_grad_fn;
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    pub fn mean(&self, axes: &[usize]) -> Tensor {
        let mut result_data = Vec::new();
        let mut count = 1.0;
        let mut reduced_shape = self.shape.clone();

        for &axis in axes {
            count *= self.shape[axis] as Scalar;
            reduced_shape[axis] = 1;
        }
        reduced_shape.retain(|&dim| dim != 1);
        if reduced_shape.is_empty() {
            reduced_shape.push(1);
        }

        let total_elements: usize = reduced_shape.iter().product();
        for i in 0..total_elements {
            let mut sum = 0.0;
            for j in 0..(self.shape.iter().product::<usize>() / total_elements) {
                let index = i + j * total_elements;
                sum += self.storage.data[index];
            }
            result_data.push(sum / count);
        }
        let strides = Tensor::compute_strides(&reduced_shape);

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: reduced_shape,
            strides,
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(MeanGradFn::new(axes.to_vec(), self.shape.clone())))
            } else {
                None
            },
            parents: if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![]
            },
            requires_grad: self.requires_grad,
        }
        .into()
    }

    pub fn mean_scalar(&self) -> Tensor {
        let total_elements: usize = self.shape.iter().product();
        let sum: Scalar = self.storage.data.iter().cloned().sum();
        let mean = sum / (total_elements as Scalar);

        InternalTensor {
            storage: Rc::new(Storage::new(vec![mean])),
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(MeanGradFn::new(
                    (0..self.shape.len()).collect(),
                    self.shape.clone(),
                )))
            } else {
                None
            },
            parents: if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![]
            },
            requires_grad: self.requires_grad,
        }
        .into()
    }

    pub fn norm(&self) -> Tensor {
        self.square().sum().sqrt()
    }

    /// Slices the tensor along the specified axis
    /// and returns a new tensor containing the slice.
    /// # Arguments
    /// * `axis` - The axis along which to slice the tensor.
    /// * `start` - The starting index of the slice (inclusive).
    /// * `len` - The length of the slice.
    /// # Returns
    /// A new tensor containing the slice.
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> Tensor {
        assert!(axis < self.shape.len(), "Axis out of bounds");
        assert!(start + len <= self.shape[axis], "Invalid slice indices");

        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;

        let mut result_data = Vec::with_capacity(new_shape.iter().product());
        let mut coords = vec![0; self.shape.len()];
        for _ in 0..new_shape.iter().product() {
            coords[axis] += start;
            let idx = self.compute_flat_index(&coords);
            result_data.push(self.storage.data[idx]);
            coords[axis] -= start;
            Tensor::increment_indices(&mut coords, &new_shape);
        }

        let strides = Tensor::compute_strides(&new_shape);

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: new_shape,
            strides,
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("Slice"),
            parents: vec![],
            requires_grad: self.requires_grad,
        }
        .into()
    }

    /// Gathers elements from the tensor along specified axis using provided indices
    /// # Arguments
    /// * `axis` - The axis along which to gather elements
    /// * `indices` - A vector containing the indices to gather
    /// # Returns
    /// A new tensor containing the gathered elements
    pub fn gather(&self, axis: usize, indices: &[usize]) -> Tensor {
        let ndim = self.shape.len();
        assert!(axis < ndim);

        let axis_dim = self.shape[axis];
        for &idx in indices {
            assert!(idx < axis_dim, "gather index out of bounds");
        }

        // Output shape
        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();

        let out_strides = Tensor::compute_strides(&out_shape);
        let total: usize = out_shape.iter().product();

        let in_strides = &self.strides;

        let mut out_data = Vec::with_capacity(total);

        let mut coord = vec![0usize; ndim];
        let mut in_offset = self.offset;

        for _ in 0..total {
            let gather_idx = indices[coord[axis]];
            let src_offset = in_offset + gather_idx * in_strides[axis];

            out_data.push(self.storage.data[src_offset]);

            // increment output index
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if d != axis {
                    in_offset += in_strides[d];
                }

                if coord[d] < out_shape[d] {
                    break;
                }

                coord[d] = 0;
                if d != axis {
                    in_offset -= in_strides[d] * out_shape[d];
                }
            }
        }

        InternalTensor {
            storage: Rc::new(Storage::new(out_data)),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("Gather"),
            parents: vec![],
            requires_grad: self.requires_grad,
        }
        .into()
    }

    /// Computes the indices of the maximum value in the tensor
    /// # Arguments
    /// * `axis` - The axis along which to compute the argmax
    /// # Returns
    /// A vector containing the indices of the maximum values along the specified axis
    /// # Example
    /// ```rust
    /// use nn_rs::linalg::tensor::Tensor;
    /// let tensor = Tensor::new(vec![4.0, 3.0, 6.0, 1.0, 5.0, 2.0], &vec![2, 3]);
    /// // 4 3 6
    /// // 1 5 2
    /// assert_eq!(tensor.argmax_axis(0), vec![0, 1, 0]); // Max values in each column are 4, 5, 6
    /// assert_eq!(tensor.argmax_axis(1), vec![2, 1]);    // Max values in each row are 6, 5
    /// ```
    pub fn argmax_axis(&self, axis: usize) -> Vec<usize> {
        assert!(axis < self.shape.len(), "Axis out of bounds");
        // find indices of max values along given axis
        let mut result_indices = Vec::new();
        let outer_dim = self.shape.iter().take(axis).product::<usize>();
        let inner_dim = self.shape.iter().skip(axis + 1).product::<usize>();
        let axis_dim = self.shape[axis];
        for outer in 0..outer_dim {
            for inner in 0..inner_dim {
                let mut max_index = 0;
                let mut max_value = Scalar::MIN;
                for axis_index in 0..axis_dim {
                    let index = outer * axis_dim * inner_dim + axis_index * inner_dim + inner;
                    let value = self.storage.data[index];
                    if value > max_value {
                        max_value = value;
                        max_index = axis_index;
                    }
                }
                result_indices.push(max_index);
            }
        }
        result_indices
    }

    /// Computes the maximum value in the tensor
    /// # Returns
    /// The maximum value
    pub fn max(&self) -> Tensor {
        let max_value = self
            .storage
            .data
            .iter()
            .cloned()
            .fold(Scalar::MIN, Scalar::max);
        InternalTensor {
            storage: Rc::new(Storage::new(vec![max_value])),
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("max"),
            parents: vec![],
            requires_grad: self.requires_grad,
        }
        .into()
    }

    /// Computes the sum of all elements in the tensor
    /// # Returns
    /// A tensor containing the sum of all elements
    pub fn sum(&self) -> Tensor {
        let sum_value: Scalar = self.storage.data.iter().cloned().sum();

        InternalTensor {
            storage: Rc::new(Storage::new(vec![sum_value])),
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(SumGradFn::new(self.shape.clone())))
            } else {
                None
            },
            parents: if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![]
            },
            requires_grad: self.requires_grad,
        }
        .into()
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(), "Axis out of bounds");
        let mut result_data = Vec::new();
        let mut reduced_shape = self.shape.clone();
        reduced_shape[axis] = 1;
        reduced_shape.retain(|&dim| dim != 1);
        if reduced_shape.is_empty() {
            reduced_shape.push(1);
        }

        let total_elements: usize = reduced_shape.iter().product();
        for i in 0..total_elements {
            let mut sum = 0.0;
            for j in 0..(self.shape.iter().product::<usize>() / total_elements) {
                let index = i + j * total_elements;
                sum += self.storage.data[index];
            }
            result_data.push(sum);
        }
        let strides = Tensor::compute_strides(&reduced_shape);

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: reduced_shape,
            strides,
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(SumAxisGradFn::new(self.shape.clone())))
            } else {
                None
            },
            parents: if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![]
            },
            requires_grad: self.requires_grad,
        }
        .into()
    }
}
