//! Basis-Composed Hypernetwork (§3.4) — synthesises field-specific projections
//! from shared bases, reducing complexity from O(F·d²) to O(M·d² + F·k).
//!
//! For each parameter type Φ ∈ {Q, K, V, FFN₁, FFN₂}:
//!   - M base matrices  B_{m,Φ}  with the same dims as the target
//!   - Single-layer MLP router  g_ψ: ℝ^k → ℝ^M
//!   - Softmax over M scores → dense coefficients
//!   - W_Φ^{(f)} = Σ_m  α_m · B_{m,Φ}
//!
//! At inference, all projections are pre-computed once at build time.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// Pre-computed field-specific projections for all FAT blocks.
pub struct FieldProjections {
    /// [num_fields, d, d]  Q projections.
    pub w_q: Tensor,
    /// [num_fields, d, d]  K projections.
    pub w_k: Tensor,
    /// [num_fields, d, d]  V projections.
    pub w_v: Tensor,
    /// [num_fields, d, d_ff]  FFN input projection W₁.
    pub w_ffn1: Tensor,
    /// [num_fields, d_ff, d]  FFN output projection W₂.
    pub w_ffn2: Tensor,
    /// [num_fields, num_fields]  field-pair interaction modulation.
    pub field_pair_w: Tensor,
}

/// Basis-Composed Hypernetwork.
#[allow(dead_code)]
pub struct BasisHypernetwork {
    num_fields: usize,
    d: usize,
    d_ff: usize,
    m: usize,
    k: usize,
    k_top: usize,

    // Base matrices  [M, d_in, d_out]
    bases_q: Tensor,
    bases_k: Tensor,
    bases_v: Tensor,
    bases_ffn1: Tensor,
    bases_ffn2: Tensor,

    // Lightweight routers (no bias, as in paper)
    router_q: Linear,
    router_k: Linear,
    router_v: Linear,
    router_ffn1: Linear,
    router_ffn2: Linear,

    // Field meta-embeddings  φ_f ∈ ℝ^k
    field_meta: Tensor,

    // Field-pair interaction modulation  w_{f_i, f_j}
    field_pair_w: Tensor,
}

impl BasisHypernetwork {
    /// Build the hypernetwork from VarBuilder.
    #[allow(non_snake_case)]
    pub fn new(
        vb: VarBuilder,
        num_fields: usize,
        d: usize,
        d_ff: usize,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<Self> {
        let hvb = vb.pp("hypernetwork");

        let bases_q = hvb.get_with_hints(
            (m, d, d),
            "bases_q",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let bases_k = hvb.get_with_hints(
            (m, d, d),
            "bases_k",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let bases_v = hvb.get_with_hints(
            (m, d, d),
            "bases_v",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let bases_ffn1 = hvb.get_with_hints(
            (m, d, d_ff),
            "bases_ffn1",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let bases_ffn2 = hvb.get_with_hints(
            (m, d_ff, d),
            "bases_ffn2",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        let router_q = linear_no_bias(k, m, hvb.pp("router_q"))?;
        let router_k = linear_no_bias(k, m, hvb.pp("router_k"))?;
        let router_v = linear_no_bias(k, m, hvb.pp("router_v"))?;
        let router_ffn1 = linear_no_bias(k, m, hvb.pp("router_ffn1"))?;
        let router_ffn2 = linear_no_bias(k, m, hvb.pp("router_ffn2"))?;

        let field_meta = hvb.get_with_hints(
            (num_fields, k),
            "field_meta",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let field_pair_w = hvb.get_with_hints(
            (num_fields, num_fields),
            "field_pair_w",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        Ok(Self {
            num_fields,
            d,
            d_ff,
            m,
            k,
            k_top,
            bases_q,
            bases_k,
            bases_v,
            bases_ffn1,
            bases_ffn2,
            router_q,
            router_k,
            router_v,
            router_ffn1,
            router_ffn2,
            field_meta,
            field_pair_w,
        })
    }

    /// Route: softmax over router scores → dense coefficients [F, M].
    fn route(&self, router: &Linear) -> Result<Tensor> {
        let scores = router.forward(&self.field_meta)?;  // [F, M]
        candle_nn::ops::softmax(&scores, 1)  // [F, M]
    }

    /// Compose:  W_f = Σ_m α_f,m · B_m   →   [F, d_in, d_out]
    ///
    /// Equivalent to einsum("fm,mid->fid", alpha, bases).
    fn compose(&self, alpha: &Tensor, bases: &Tensor) -> Result<Tensor> {
        // alpha: [F, M],  bases: [M, d_in, d_out]
        // Expand for broadcasting:
        //   α_{f,m,1,1} · B_{m,d_in,d_out} → sum over m
        let a = alpha.unsqueeze(2)?.unsqueeze(3)?;      // [F, M, 1, 1]
        let b = bases.unsqueeze(0)?;                     // [1, M, d_in, d_out]
        let weighted = a.broadcast_mul(&b)?;             // [F, M, d_in, d_out]
        weighted.sum(1)                                  // [F, d_in, d_out]
    }

    /// Pre-compute all field-specific projections.
    ///
    /// Called once at build time; the composed tensors are cached for all blocks.
    pub fn precompute_all(&self) -> Result<FieldProjections> {
        let a_q = self.route(&self.router_q)?;
        let a_k = self.route(&self.router_k)?;
        let a_v = self.route(&self.router_v)?;
        let a_1 = self.route(&self.router_ffn1)?;
        let a_2 = self.route(&self.router_ffn2)?;

        Ok(FieldProjections {
            w_q: self.compose(&a_q, &self.bases_q)?,
            w_k: self.compose(&a_k, &self.bases_k)?,
            w_v: self.compose(&a_v, &self.bases_v)?,
            w_ffn1: self.compose(&a_1, &self.bases_ffn1)?,
            w_ffn2: self.compose(&a_2, &self.bases_ffn2)?,
            field_pair_w: self.field_pair_w.clone(),
        })
    }
}
