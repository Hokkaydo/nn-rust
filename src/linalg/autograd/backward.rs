use crate::linalg::autograd::grad_fn::GradFn;
use crate::linalg::tensor_grad::{InternalTensor, Scalar, Tensor};
use std::collections::HashSet;
use std::rc::Rc;

impl InternalTensor {
    pub(crate) fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
}

impl Tensor {
    pub(crate) fn set_grad_metadata(&mut self, grad_fn: Rc<dyn GradFn>, parents: Vec<Tensor>) {
        let inner = Rc::make_mut(&mut self.0);
        inner.grad_fn = Some(grad_fn);
        inner.parents = parents;
    }
    /// Creates a new tensor with the given data and shape, with `requires_grad` set to true.
    /// # Arguments
    /// * `data` - A vector containing the tensor data.
    /// * `shape` - A slice representing the shape of the tensor.
    /// # Returns
    /// A new tensor with `requires_grad` set to true.
    pub fn with_grad(data: Vec<Scalar>, shape: &[usize]) -> Self {
        InternalTensor::new(data, shape)
            .with_requires_grad(true)
            .into()
    }

    /// Returns the gradient if it exists as an `Option`.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().as_ref().cloned()
    }

    pub fn clear_graph(&mut self) {
        let inner = Rc::make_mut(&mut self.0);
        inner.grad_fn = None;
        inner.parents.clear();
    }

    /// Performs backpropagation to compute gradients for all tensors in the computation graph
    /// that have `requires_grad` set to true.
    /// Gradients are accumulated in the `grad` field of each tensor.
    pub fn backward(self: &Self) {
        self.backward_with_options(false);
    }

    /// Performs backpropagation to compute gradients for all tensors in the computation graph
    /// that have `requires_grad` set to true.
    /// # Arguments
    /// * `retain_graph` If true, retains the computation graph after backward pass,
    /// else clears gradients of intermediate tensors.
    pub fn backward_with_options(self: &Self, fresh_graph: bool) {
        assert!(self.requires_grad);

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        if fresh_graph {
            for t in &mut topo {
                t.clear_graph();
            }
        }

        *self.grad.borrow_mut() = Some(Tensor::ones(&self.shape));

        for t in topo.into_iter().rev() {
            let grad_out = match &*t.grad.borrow() {
                Some(g) => g.clone(),
                None => continue,
            };

            let grad_fn = match &t.grad_fn {
                Some(f) => f,
                None => continue,
            };
            // println!("gradfn {:?}", grad_fn.type_name());
            // println!("tensor {:?}", t);

            let parent_grads = grad_fn.apply(&t, &grad_out);

            for (parent, g) in t.parents.iter().zip(parent_grads) {
                if !parent.requires_grad {
                    continue;
                }

                let mut parent_grad = parent.grad.borrow_mut();
                match &mut *parent_grad {
                    Some(existing) => {
                        *existing = &*existing + &g;
                    }
                    None => {
                        *parent_grad = Some(g);
                    }
                }
            }
        }
    }

    /// Sums the tensor to match the specified shape by summing over dimensions where the target shape has size 1.
    /// # Arguments
    /// * `shape` - The target shape to sum to.
    /// # Returns
    /// A new tensor summed to the specified shape.
    pub fn sum_to_shape(&self, shape: &[usize]) -> Tensor {
        let mut result = self.clone();
        for (i, &dim) in Tensor::reduce_shape(self.shape()).iter().enumerate() {
            if shape[i] == 1 && dim != 1 {
                result = result.sum_axis(i);
            }
        }
        result.reshape(shape)
    }

    /// Expands the tensor by repeating its data along a new dimension.
    /// # Arguments
    /// * `axis` - The axis along which to expand the tensor.
    /// * `size` - The size of the new dimension.
    /// # Returns
    /// A new tensor with the expanded dimension.
    pub fn expand_dim(&self, axis: usize, size: usize) -> Tensor {
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, size); // insert the repeated dimension

        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        let old_strides = Tensor::compute_strides(&self.shape());

        // Iterate over all coordinates of the new tensor
        let mut indices = vec![0; new_shape.len()];
        for _ in 0..new_data.capacity() {
            // Map new indices to old tensor indices
            let mut old_indices = indices.clone();
            old_indices.remove(axis); // remove the repeated axis
            let flat_idx: usize = old_indices
                .iter()
                .zip(&old_strides)
                .map(|(i, s)| i * s)
                .sum();
            new_data.push(self.storage.data[flat_idx]);

            // Increment indices
            for d in (0..new_shape.len()).rev() {
                indices[d] += 1;
                if indices[d] < new_shape[d] {
                    break;
                } else {
                    indices[d] = 0;
                }
            }
        }

        Tensor::new(new_data, &new_shape)
    }
}

fn build_topo(t: &Tensor, visited: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
    let id = Rc::as_ptr(&t.0) as usize;
    if visited.contains(&id) {
        return;
    }
    visited.insert(id);

    for p in &t.parents {
        build_topo(p, visited, out);
    }

    out.push(t.clone());
}
