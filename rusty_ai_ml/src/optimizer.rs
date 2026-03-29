//! First-order optimization: SGD and Adam on [`rusty_ai_autograd::Variable`] parameters.
//!
//! TODO: AdamW, learning-rate schedulers, gradient clipping.

use std::rc::Rc;

use rusty_ai_autograd::Variable;
use rusty_ai_core::{add, div, mul, sqrt, sub, Tensor, TensorError};

/// Stochastic gradient descent: `θ <- θ - lr * ∇θ`.
pub struct Sgd {
    pub lr: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Applies gradients to each parameter that has a non-`None` [`Variable::grad`].
    pub fn step(&self, params: &[Rc<Variable>]) -> Result<(), TensorError> {
        for p in params {
            if let Some(g) = p.grad() {
                let upd = mul(&g, &Tensor::scalar(self.lr))?;
                let new_data = sub(&p.data(), &upd)?;
                p.set_data(new_data);
            }
        }
        Ok(())
    }
}

/// Adam optimizer with bias-corrected first and second moments (Kingma & Ba).
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u64,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// One update step; aligns moment buffers with `params` by index on first use.
    pub fn step(&mut self, params: &[Rc<Variable>]) -> Result<(), TensorError> {
        self.t += 1;
        let t = self.t as f32;
        if self.m.len() != params.len() {
            self.m = (0..params.len()).map(|_| None).collect();
            self.v = (0..params.len()).map(|_| None).collect();
        }

        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);

        for (i, p) in params.iter().enumerate() {
            let Some(g) = p.grad() else {
                continue;
            };
            let m_prev = self.m[i].take().unwrap_or_else(|| {
                Tensor::zeros(g.shape(), rusty_ai_core::DType::F32).expect("zeros")
            });
            let v_prev = self.v[i].take().unwrap_or_else(|| {
                Tensor::zeros(g.shape(), rusty_ai_core::DType::F32).expect("zeros")
            });

            let m_new = add(
                &mul(&m_prev, &Tensor::scalar(self.beta1))?,
                &mul(&g, &Tensor::scalar(1.0 - self.beta1))?,
            )?;
            let g2 = mul(&g, &g)?;
            let v_new = add(
                &mul(&v_prev, &Tensor::scalar(self.beta2))?,
                &mul(&g2, &Tensor::scalar(1.0 - self.beta2))?,
            )?;

            let m_hat = div(&m_new, &Tensor::scalar(b1t))?;
            let v_hat = div(&v_new, &Tensor::scalar(b2t))?;
            let denom = add(&sqrt(&v_hat), &Tensor::scalar(self.eps))?;
            let step = div(&mul(&m_hat, &Tensor::scalar(self.lr))?, &denom)?;
            let new_data = sub(&p.data(), &step)?;

            self.m[i] = Some(m_new);
            self.v[i] = Some(v_new);
            p.set_data(new_data);
        }
        Ok(())
    }
}
