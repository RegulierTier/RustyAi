//! [`Variable`]: a node in the computation graph with optional gradients.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use rusty_ai_core::{
    add, matmul, mse, mul, relu, sub, sum_axis_0, transpose_2d, Tensor, TensorError,
};

/// A tensor value participating in autograd, plus the operation that produced it (if any).
///
/// - **Data** live in a [`RefCell`] so optimizers can replace weights after `backward`.
/// - **Gradients** are stored when `backward` runs; call [`Variable::zero_grad`] before each
///   new forward pass if you reuse the same parameter nodes.
pub struct Variable {
    storage: RefCell<Tensor>,
    grad: RefCell<Option<Tensor>>,
    op: Op,
}

/// Discriminated backward rule: each variant knows how to route `grad_output` to parents.
#[derive(Clone)]
enum Op {
    /// Constant or parameter with no parents.
    Leaf,
    /// Elementwise/broadcast addition: gradient w.r.t. both inputs equals `grad_output`.
    Add(Rc<Variable>, Rc<Variable>),
    /// `output = x + bias` with `x` shaped `(batch, n)` and `bias` `(1, n)` broadcast.
    /// Bias gradient is the sum of `grad_output` over batch (axis 0).
    BiasAdd(Rc<Variable>, Rc<Variable>),
    /// Matrix multiply: uses standard rules `dA = dC @ B^T`, `dB = A^T @ dC`.
    MatMul(Rc<Variable>, Rc<Variable>),
    /// ReLU: gradient passes through where input was positive.
    Relu(Rc<Variable>),
    /// Mean squared error against a **constant** target tensor (no grad through `target`).
    Mse(Rc<Variable>, Tensor),
}

impl Variable {
    /// Creates a leaf node (trainable parameter or input batch).
    pub fn leaf(data: Tensor) -> Rc<Self> {
        Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Leaf,
        })
    }

    /// Borrows the forward tensor immutably.
    pub fn data(&self) -> Ref<'_, Tensor> {
        self.storage.borrow()
    }

    /// Borrows the forward tensor mutably (rare; prefer optimizer `set_data` patterns).
    pub fn data_mut(&self) -> RefMut<'_, Tensor> {
        self.storage.borrow_mut()
    }

    /// Replaces the forward tensor (e.g. after an optimizer step).
    pub fn set_data(&self, t: Tensor) {
        *self.storage.borrow_mut() = t;
    }

    /// Broadcast addition: delegates to `rusty_ai_core::add` for the forward value.
    pub fn add(a: &Rc<Variable>, b: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = add(&a.data(), &b.data())?;
        if !crate::grad_enabled() {
            return Ok(Rc::new(Self {
                storage: RefCell::new(data),
                grad: RefCell::new(None),
                op: Op::Leaf,
            }));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Add(Rc::clone(a), Rc::clone(b)),
        }))
    }

    /// Adds a row bias: `x` is `(batch, n)`, `bias` is `(1, n)`.
    pub fn bias_add(x: &Rc<Variable>, bias: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = add(&x.data(), &bias.data())?;
        if !crate::grad_enabled() {
            return Ok(Rc::new(Self {
                storage: RefCell::new(data),
                grad: RefCell::new(None),
                op: Op::Leaf,
            }));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::BiasAdd(Rc::clone(x), Rc::clone(bias)),
        }))
    }

    /// Batched/2D matrix multiplication of forward tensors.
    pub fn matmul(a: &Rc<Variable>, b: &Rc<Variable>) -> Result<Rc<Self>, TensorError> {
        let data = matmul(&a.data(), &b.data())?;
        if !crate::grad_enabled() {
            return Ok(Rc::new(Self {
                storage: RefCell::new(data),
                grad: RefCell::new(None),
                op: Op::Leaf,
            }));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::MatMul(Rc::clone(a), Rc::clone(b)),
        }))
    }

    /// ReLU activation (non-differentiable at 0; subgradient 0 is used).
    pub fn relu(v: &Rc<Variable>) -> Rc<Self> {
        let data = relu(&v.data());
        if !crate::grad_enabled() {
            return Rc::new(Self {
                storage: RefCell::new(data),
                grad: RefCell::new(None),
                op: Op::Leaf,
            });
        }
        Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Relu(Rc::clone(v)),
        })
    }

    /// Mean squared error vs. a fixed target; forward is a 0-D scalar loss tensor.
    pub fn mse(pred: &Rc<Variable>, target: &Tensor) -> Result<Rc<Self>, TensorError> {
        let data = mse(&pred.data(), target)?;
        if !crate::grad_enabled() {
            return Ok(Rc::new(Self {
                storage: RefCell::new(data),
                grad: RefCell::new(None),
                op: Op::Leaf,
            }));
        }
        Ok(Rc::new(Self {
            storage: RefCell::new(data),
            grad: RefCell::new(None),
            op: Op::Mse(Rc::clone(pred), target.clone()),
        }))
    }

    /// Gradient tensor after [`backward`], if any was computed.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().clone()
    }

    /// Clears accumulated gradient before a new forward/backward cycle.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }
}

/// Adds `g` into `v.grad` (summing if the node receives gradient from multiple uses).
fn acc_grad(v: &Variable, g: &Tensor) -> Result<(), TensorError> {
    let mut cell = v.grad.borrow_mut();
    match &mut *cell {
        Some(acc) => {
            *acc = add(acc, g)?;
        }
        None => {
            *cell = Some(g.clone());
        }
    }
    Ok(())
}

/// Runs backpropagation from a **scalar** loss: upstream gradient is `1`.
pub fn backward(loss: &Rc<Variable>) -> Result<(), TensorError> {
    let g = Tensor::scalar(1.0);
    backward_grad(loss, &g)
}

fn backward_grad(v: &Rc<Variable>, grad: &Tensor) -> Result<(), TensorError> {
    acc_grad(v, grad)?;

    match &v.op {
        Op::Leaf => Ok(()),
        Op::Add(a, b) => {
            backward_grad(a, grad)?;
            backward_grad(b, grad)?;
            Ok(())
        }
        Op::BiasAdd(x, b) => {
            backward_grad(x, grad)?;
            let gb = sum_axis_0(grad)?;
            backward_grad(b, &gb)?;
            Ok(())
        }
        Op::MatMul(a, b) => {
            let ga = matmul(grad, &transpose_2d(&b.data())?)?;
            let gb = matmul(&transpose_2d(&a.data())?, grad)?;
            backward_grad(a, &ga)?;
            backward_grad(b, &gb)?;
            Ok(())
        }
        Op::Relu(x) => {
            let mask = relu_mask(&x.data());
            let gx = mul(grad, &mask)?;
            backward_grad(x, &gx)?;
            Ok(())
        }
        Op::Mse(pred, target) => {
            let n = pred.data().numel().max(1) as f32;
            let diff = sub(&pred.data(), target)?;
            // Chain rule: d(MSE)/d(pred) = (2/n) * (pred - target) * d(loss)/d(scalar).
            let g0 = grad.data().first().copied().unwrap_or(1.0);
            let scale = mul(&diff, &Tensor::scalar(2.0 / n * g0))?;
            backward_grad(pred, &scale)?;
            Ok(())
        }
    }
}

/// Binary mask: 1 where `x > 0`, else 0 (subgradient of ReLU).
fn relu_mask(x: &Tensor) -> Tensor {
    let mut out = x.data().to_vec();
    for v in &mut out {
        if *v > 0.0 {
            *v = 1.0;
        } else {
            *v = 0.0;
        }
    }
    Tensor::from_vec(out, x.shape().to_vec()).expect("same shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_backward_runs() {
        let x = Variable::leaf(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let w = Variable::leaf(Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]).unwrap());
        let h = Variable::matmul(&x, &w).unwrap();
        let y = Variable::relu(&h);
        let target = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let loss = Variable::mse(&y, &target).unwrap();
        backward(&loss).unwrap();
        assert!(w.grad().is_some());
        assert!(x.grad().is_some());
    }
}
