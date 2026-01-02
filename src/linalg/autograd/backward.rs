use crate::linalg::tensor_grad::Tensor;

impl Tensor {
    /// Sets the gradient of the tensor_old to zero.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Marks the tensor_old to require gradient computation.
    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Returns the gradient tensor_old if it exists as an `Option`.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().as_ref().map(|g| (**g).clone())
    }
}
