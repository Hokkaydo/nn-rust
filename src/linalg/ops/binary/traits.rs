use crate::linalg::ops::binary::kernels;
use crate::linalg::tensor::{Scalar, Tensor};
use std::ops::{Add, Div, Mul, Sub};

/// Addition implementations
/// --------------------------------------
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        kernels::add_tt(self, other)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        &self + &other
    }
}

impl Add<Scalar> for &Tensor {
    type Output = Tensor;
    fn add(self, other: Scalar) -> Tensor {
        kernels::add_ts(self, other)
    }
}

impl Add<&Tensor> for Scalar {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        kernels::add_ts(other, self)
    }
}

impl Add<Scalar> for Tensor {
    type Output = Tensor;
    fn add(self, other: Scalar) -> Tensor {
        &self + other
    }
}

impl Add<Tensor> for Scalar {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        self + &other
    }
}

/// Subtraction implementations
/// --------------------------------------
impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        kernels::sub_tt(self, other)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        &self - &other
    }
}

impl Sub<Scalar> for &Tensor {
    type Output = Tensor;
    fn sub(self, other: Scalar) -> Tensor {
        kernels::sub_ts(self, other)
    }
}

impl Sub<&Tensor> for Scalar {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        kernels::sub_st(self, other)
    }
}

impl Sub<Scalar> for Tensor {
    type Output = Tensor;
    fn sub(self, other: Scalar) -> Tensor {
        &self - other
    }
}

impl Sub<Tensor> for Scalar {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        self - &other
    }
}

/// Scalar & Element-wise multiplication implementations
/// --------------------------------------
impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        kernels::mul_tt_ews(self, other)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        &self * &other
    }
}

impl Mul<Scalar> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: Scalar) -> Tensor {
        kernels::mul_ts(self, other)
    }
}

impl Mul<&Tensor> for Scalar {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        kernels::mul_ts(other, self)
    }
}

impl Mul<Scalar> for Tensor {
    type Output = Tensor;
    fn mul(self, other: Scalar) -> Tensor {
        &self * other
    }
}

impl Mul<Tensor> for Scalar {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        self * &other
    }
}

/// Scalar & Element-wise division implementations
/// --------------------------------------
impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        kernels::div_tt_ews(self, other)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        &self / &other
    }
}

impl Div<Scalar> for &Tensor {
    type Output = Tensor;
    fn div(self, other: Scalar) -> Tensor {
        kernels::div_ts(self, other)
    }
}

impl Div<&Tensor> for Scalar {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        kernels::div_st(self, other)
    }
}

impl Div<Scalar> for Tensor {
    type Output = Tensor;
    fn div(self, other: Scalar) -> Tensor {
        &self / other
    }
}

impl Div<Tensor> for Scalar {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        self / &other
    }
}
