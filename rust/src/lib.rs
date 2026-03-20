use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

const NEG_INF: f64 = f64::NEG_INFINITY;

fn extract_ascii_base(key: &Bound<'_, PyAny>) -> PyResult<u8> {
    let s: String = key.extract()?;
    let bytes = s.as_bytes();
    if bytes.len() != 1 {
        return Err(PyValueError::new_err(
            "score_matrix keys must be single characters",
        ));
    }
    Ok(bytes[0].to_ascii_uppercase())
}

fn build_score_table(score_matrix: &Bound<'_, PyDict>) -> PyResult<Vec<f64>> {
    let mut table = vec![0.0_f64; 256 * 256];
    for (k, inner) in score_matrix.iter() {
        let row_key = extract_ascii_base(&k)?;
        let inner_dict = inner.downcast::<PyDict>()?;
        for (kk, vv) in inner_dict.iter() {
            let col_key = extract_ascii_base(&kk)?;
            let score: f64 = vv.extract()?;
            table[(row_key as usize) * 256 + (col_key as usize)] = score;
        }
    }
    Ok(table)
}

#[inline]
fn table_score(table: &[f64], a: u8, b: u8) -> f64 {
    table[(a as usize) * 256 + (b as usize)]
}

#[inline]
fn resolve_gap_costs(gap_open: f64, gap_extend: f64, enable_gap_close_penalty: bool) -> (f64, f64) {
    if !enable_gap_close_penalty {
        return (gap_open, 0.0);
    }
    let effective_gap_open = gap_extend + (gap_open - gap_extend) / 2.0;
    let gap_close_penalty = (gap_open - gap_extend) / 2.0;
    (effective_gap_open, gap_close_penalty)
}

#[pyclass(module = "sgad_rust_native")]
#[derive(Clone)]
struct RustScoreScaler {
    decay_exponent: f64,
    temperature: f64,
}

impl RustScoreScaler {
    fn checked(decay_exponent: f64, temperature: f64) -> PyResult<Self> {
        if temperature <= 0.0 {
            return Err(PyValueError::new_err("temperature must be > 0"));
        }
        Ok(Self {
            decay_exponent,
            temperature,
        })
    }

    #[inline]
    fn factor(
        &self,
        seq1_left_idx: isize,
        seq1_right_idx: isize,
        seq2_left_idx: isize,
        seq2_right_idx: isize,
        seq1_left_free: bool,
        seq1_right_free: bool,
        seq2_left_free: bool,
        seq2_right_free: bool,
    ) -> f64 {
        let mut sum = 0.0;
        let de = self.decay_exponent;
        let temp = self.temperature;

        if seq1_left_free {
            sum += 1.0 / (((seq1_left_idx as f64 + 1.0) / temp).powf(de));
        }
        if seq1_right_free {
            sum += 1.0 / (((seq1_right_idx as f64 + 1.0) / temp).powf(de));
        }
        if seq2_left_free {
            sum += 1.0 / (((seq2_left_idx as f64 + 1.0) / temp).powf(de));
        }
        if seq2_right_free {
            sum += 1.0 / (((seq2_right_idx as f64 + 1.0) / temp).powf(de));
        }

        if sum == 0.0 {
            1.0
        } else {
            sum
        }
    }
}

#[pymethods]
impl RustScoreScaler {
    #[new]
    #[pyo3(signature = (decay_exponent=1.0, temperature=1.0))]
    fn new(decay_exponent: f64, temperature: f64) -> PyResult<Self> {
        Self::checked(decay_exponent, temperature)
    }

    #[pyo3(signature = (
        seq1_left_idx,
        seq1_right_idx,
        seq2_left_idx,
        seq2_right_idx,
        seq1_left_free,
        seq1_right_free,
        seq2_left_free,
        seq2_right_free
    ))]
    fn __call__(
        &self,
        seq1_left_idx: isize,
        seq1_right_idx: isize,
        seq2_left_idx: isize,
        seq2_right_idx: isize,
        seq1_left_free: bool,
        seq1_right_free: bool,
        seq2_left_free: bool,
        seq2_right_free: bool,
    ) -> f64 {
        self.factor(
            seq1_left_idx,
            seq1_right_idx,
            seq2_left_idx,
            seq2_right_idx,
            seq1_left_free,
            seq1_right_free,
            seq2_left_free,
            seq2_right_free,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (decay_exponent=1.0, temperature=1.0))]
fn make_rust_score_scaler(decay_exponent: f64, temperature: f64) -> PyResult<RustScoreScaler> {
    RustScoreScaler::checked(decay_exponent, temperature)
}

fn needleman_wunsch_core(
    seq1: &str,
    seq2: &str,
    table: &[f64],
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    enable_gap_close_penalty: bool,
    rust_scaler: Option<&RustScoreScaler>,
) -> PyResult<(String, String, f64)> {
    let a: Vec<u8> = seq1
        .as_bytes()
        .iter()
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let b: Vec<u8> = seq2
        .as_bytes()
        .iter()
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let n = a.len();
    let m = b.len();
    let row_stride = m + 1;
    let nm = (n + 1) * row_stride;
    let plane0 = 0;
    let plane1 = nm;
    let plane2 = 2 * nm;
    let (effective_gap_open, gap_close_penalty) =
        resolve_gap_costs(gap_open, gap_extend, enable_gap_close_penalty);

    let mut dp = vec![NEG_INF; 3 * nm];
    let mut ptr_state = vec![u8::MAX; 3 * nm];

    let mut factor0 = vec![1.0_f64; nm];
    let mut factor1 = vec![1.0_f64; nm];
    let mut factor2 = vec![1.0_f64; nm];

    if let Some(scaler) = rust_scaler {
        let de = scaler.decay_exponent;
        let temp = scaler.temperature;

        let mut s1_left_mask0 = vec![0.0_f64; n + 1];
        let mut s1_left_mask1 = vec![0.0_f64; n + 1];
        let mut s1_right = vec![0.0_f64; n + 1];
        let mut s2_left_mask0 = vec![0.0_f64; m + 1];
        let mut s2_left_mask2 = vec![0.0_f64; m + 1];
        let mut s2_right = vec![0.0_f64; m + 1];

        if seq1_left_free {
            for i in 0..=n {
                let base0 = i as f64 / temp;
                s1_left_mask0[i] = 1.0 / base0.powf(de);
                let base1 = (i + 1) as f64 / temp;
                s1_left_mask1[i] = 1.0 / base1.powf(de);
            }
        }
        if seq1_right_free {
            for i in 0..=n {
                let base = (n - i + 1) as f64 / temp;
                s1_right[i] = 1.0 / base.powf(de);
            }
        }
        if seq2_left_free {
            for j in 0..=m {
                let base0 = j as f64 / temp;
                s2_left_mask0[j] = 1.0 / base0.powf(de);
                let base2 = (j + 1) as f64 / temp;
                s2_left_mask2[j] = 1.0 / base2.powf(de);
            }
        }
        if seq2_right_free {
            for j in 0..=m {
                let base = (m - j + 1) as f64 / temp;
                s2_right[j] = 1.0 / base.powf(de);
            }
        }

        for i in 0..=n {
            let row_start = i * row_stride;
            let s1_l0 = s1_left_mask0[i];
            let s1_l1 = s1_left_mask1[i];
            let s1_r = s1_right[i];

            for j in 0..=m {
                let at = row_start + j;
                let s2_l0 = s2_left_mask0[j];
                let s2_l2 = s2_left_mask2[j];
                let s2_r = s2_right[j];

                let f0 = s1_l0 + s1_r + s2_l0 + s2_r;
                let f1 = s1_l1 + s1_r + s2_l0 + s2_r;
                let f2 = s1_l0 + s1_r + s2_l2 + s2_r;

                factor0[at] = if f0 == 0.0 { 1.0 } else { f0 };
                factor1[at] = if f1 == 0.0 { 1.0 } else { f1 };
                factor2[at] = if f2 == 0.0 { 1.0 } else { f2 };
            }
        }
    }

    dp[plane0] = 0.0;
    ptr_state[plane0] = 0;

    for i in 0..=n {
        let row_start = i * row_stride;
        let prev_row_start = if i > 0 { (i - 1) * row_stride } else { 0 };

        for j in 0..=m {
            if i == 0 && j == 0 {
                continue;
            }

            let at = row_start + j;

            if i > 0 && j > 0 {
                let prev_at = prev_row_start + (j - 1);
                let sub = table_score(table, a[i - 1], b[j - 1]) * factor0[at];
                let close_from_state1 = if seq1_left_free && i == 1 {
                    0.0
                } else {
                    gap_close_penalty * factor1[prev_at]
                };
                let close_from_state2 = if seq2_left_free && j == 1 {
                    0.0
                } else {
                    gap_close_penalty * factor2[prev_at]
                };

                let mut best = NEG_INF;
                let mut best_prev = u8::MAX;

                let prev0 = dp[plane0 + prev_at];
                if prev0 != NEG_INF {
                    let cand = prev0 + sub;
                    if cand > best {
                        best = cand;
                        best_prev = 0;
                    }
                }

                let prev1 = dp[plane1 + prev_at];
                if prev1 != NEG_INF {
                    let cand = prev1 + sub + close_from_state1;
                    if cand > best {
                        best = cand;
                        best_prev = 1;
                    }
                }

                let prev2 = dp[plane2 + prev_at];
                if prev2 != NEG_INF {
                    let cand = prev2 + sub + close_from_state2;
                    if cand > best {
                        best = cand;
                        best_prev = 2;
                    }
                }

                if best_prev != u8::MAX {
                    let at0 = plane0 + at;
                    dp[at0] = best;
                    ptr_state[at0] = best_prev;
                }
            }

            if j > 0 {
                let prev_at = row_start + (j - 1);
                let penalize_gap1 = !((i == 0 && seq1_left_free) || (i == n && seq1_right_free));
                let gap1_factor = factor1[at];
                let open1 = if penalize_gap1 {
                    effective_gap_open * gap1_factor
                } else {
                    0.0
                };
                let ext1 = if penalize_gap1 {
                    gap_extend * gap1_factor
                } else {
                    0.0
                };
                let close_from_state2 = if seq2_left_free && j == 1 {
                    0.0
                } else {
                    gap_close_penalty * factor2[prev_at]
                };

                let mut best = NEG_INF;
                let mut best_prev = u8::MAX;

                let prev0 = dp[plane0 + prev_at];
                if prev0 != NEG_INF {
                    let cand = prev0 + open1;
                    if cand > best {
                        best = cand;
                        best_prev = 0;
                    }
                }

                let prev1 = dp[plane1 + prev_at];
                if prev1 != NEG_INF {
                    let cand = prev1 + ext1;
                    if cand > best {
                        best = cand;
                        best_prev = 1;
                    }
                }

                let prev2 = dp[plane2 + prev_at];
                if prev2 != NEG_INF {
                    let cand = prev2 + open1 + close_from_state2;
                    if cand > best {
                        best = cand;
                        best_prev = 2;
                    }
                }

                if best_prev != u8::MAX {
                    let at1 = plane1 + at;
                    dp[at1] = best;
                    ptr_state[at1] = best_prev;
                }
            }

            if i > 0 {
                let prev_at = prev_row_start + j;
                let penalize_gap2 = !((j == 0 && seq2_left_free) || (j == m && seq2_right_free));
                let gap2_factor = factor2[at];
                let open2 = if penalize_gap2 {
                    effective_gap_open * gap2_factor
                } else {
                    0.0
                };
                let ext2 = if penalize_gap2 {
                    gap_extend * gap2_factor
                } else {
                    0.0
                };
                let close_from_state1 = if seq1_left_free && i == 1 {
                    0.0
                } else {
                    gap_close_penalty * factor1[prev_at]
                };

                let mut best = NEG_INF;
                let mut best_prev = u8::MAX;

                let prev0 = dp[plane0 + prev_at];
                if prev0 != NEG_INF {
                    let cand = prev0 + open2;
                    if cand > best {
                        best = cand;
                        best_prev = 0;
                    }
                }

                let prev1 = dp[plane1 + prev_at];
                if prev1 != NEG_INF {
                    let cand = prev1 + open2 + close_from_state1;
                    if cand > best {
                        best = cand;
                        best_prev = 1;
                    }
                }

                let prev2 = dp[plane2 + prev_at];
                if prev2 != NEG_INF {
                    let cand = prev2 + ext2;
                    if cand > best {
                        best = cand;
                        best_prev = 2;
                    }
                }

                if best_prev != u8::MAX {
                    let at2 = plane2 + at;
                    dp[at2] = best;
                    ptr_state[at2] = best_prev;
                }
            }
        }
    }

    let end_at = n * row_stride + m;
    let mut end_scores = [NEG_INF; 3];
    end_scores[0] = dp[plane0 + end_at];

    let mut end1 = dp[plane1 + end_at];
    if end1 != NEG_INF && gap_close_penalty != 0.0 && !seq1_right_free {
        end1 += gap_close_penalty * factor1[end_at];
    }
    end_scores[1] = end1;

    let mut end2 = dp[plane2 + end_at];
    if end2 != NEG_INF && gap_close_penalty != 0.0 && !seq2_right_free {
        end2 += gap_close_penalty * factor2[end_at];
    }
    end_scores[2] = end2;

    let mut best_state = 0_usize;
    let mut best_score = end_scores[0];
    for s in 1..3 {
        if end_scores[s] > best_score {
            best_score = end_scores[s];
            best_state = s;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut state = best_state;
    let mut out_a: Vec<u8> = Vec::with_capacity(n + m);
    let mut out_b: Vec<u8> = Vec::with_capacity(n + m);

    while i > 0 || j > 0 {
        let cell_i = i;
        let cell_j = j;
        match state {
            0 => {
                out_a.push(a[cell_i - 1]);
                out_b.push(b[cell_j - 1]);
                i -= 1;
                j -= 1;
            }
            1 => {
                out_a.push(b'-');
                out_b.push(b[cell_j - 1]);
                j -= 1;
            }
            2 => {
                out_a.push(a[cell_i - 1]);
                out_b.push(b'-');
                i -= 1;
            }
            _ => {
                return Err(PyValueError::new_err(
                    "invalid state encountered during traceback",
                ));
            }
        }

        let at = cell_i * row_stride + cell_j;
        let prev_state = ptr_state[state * nm + at];
        if prev_state == u8::MAX {
            return Err(PyValueError::new_err(
                "unset pointer encountered during traceback",
            ));
        }
        state = prev_state as usize;
    }

    out_a.reverse();
    out_b.reverse();

    // Safety: outputs contain only ASCII bases copied from input bytes and '-'.
    let aligned_a = unsafe { String::from_utf8_unchecked(out_a) };
    // Safety: outputs contain only ASCII bases copied from input bytes and '-'.
    let aligned_b = unsafe { String::from_utf8_unchecked(out_b) };

    Ok((aligned_a, aligned_b, best_score))
}

#[doc(hidden)]
pub fn bench_needleman_wunsch_2d_no_scaler(
    seq1: &str,
    seq2: &str,
    table: &[f64],
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    enable_gap_close_penalty: bool,
) -> (String, String, f64) {
    needleman_wunsch_core(
        seq1,
        seq2,
        table,
        gap_open,
        gap_extend,
        seq1_left_free,
        seq1_right_free,
        seq2_left_free,
        seq2_right_free,
        enable_gap_close_penalty,
        None,
    )
    .expect("2D Needleman-Wunsch benchmark helper should not fail")
}

#[doc(hidden)]
pub fn bench_needleman_wunsch_2d_with_scaler(
    seq1: &str,
    seq2: &str,
    table: &[f64],
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    enable_gap_close_penalty: bool,
    decay_exponent: f64,
    temperature: f64,
) -> (String, String, f64) {
    let scaler = RustScoreScaler::checked(decay_exponent, temperature)
        .expect("benchmark scaler must use valid parameters");
    needleman_wunsch_core(
        seq1,
        seq2,
        table,
        gap_open,
        gap_extend,
        seq1_left_free,
        seq1_right_free,
        seq2_left_free,
        seq2_right_free,
        enable_gap_close_penalty,
        Some(&scaler),
    )
    .expect("2D Needleman-Wunsch benchmark helper should not fail")
}

#[pyfunction]
#[pyo3(signature = (
    seq1,
    seq2,
    *,
    score_matrix,
    gap_open=-5.0,
    gap_extend=-1.0,
    seq1_left_free=false,
    seq1_right_free=false,
    seq2_left_free=false,
    seq2_right_free=false,
    enable_gap_close_penalty=true,
    score_scaler_fn=None
))]
fn needleman_wunsch(
    py: Python<'_>,
    seq1: &str,
    seq2: &str,
    score_matrix: &Bound<'_, PyDict>,
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    enable_gap_close_penalty: bool,
    score_scaler_fn: Option<Py<RustScoreScaler>>,
) -> PyResult<(String, String, f64)> {
    let table = build_score_table(score_matrix)?;
    let rust_scaler = score_scaler_fn.map(|s| s.borrow(py).clone());
    needleman_wunsch_core(
        seq1,
        seq2,
        &table,
        gap_open,
        gap_extend,
        seq1_left_free,
        seq1_right_free,
        seq2_left_free,
        seq2_right_free,
        enable_gap_close_penalty,
        rust_scaler.as_ref(),
    )
}

#[pyfunction]
#[pyo3(signature = (
    seq_pairs,
    *,
    score_matrix,
    gap_open=-5.0,
    gap_extend=-1.0,
    seq1_left_free=false,
    seq1_right_free=false,
    seq2_left_free=false,
    seq2_right_free=false,
    enable_gap_close_penalty=true,
    score_scaler_fn=None
))]
fn needleman_wunsch_batch(
    py: Python<'_>,
    seq_pairs: Vec<(String, String)>,
    score_matrix: &Bound<'_, PyDict>,
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    enable_gap_close_penalty: bool,
    score_scaler_fn: Option<Py<RustScoreScaler>>,
) -> PyResult<Vec<(String, String, f64)>> {
    let table = build_score_table(score_matrix)?;
    let rust_scaler = score_scaler_fn.map(|s| s.borrow(py).clone());
    let mut out: Vec<(String, String, f64)> = Vec::with_capacity(seq_pairs.len());

    for (seq1, seq2) in seq_pairs {
        out.push(needleman_wunsch_core(
            &seq1,
            &seq2,
            &table,
            gap_open,
            gap_extend,
            seq1_left_free,
            seq1_right_free,
            seq2_left_free,
            seq2_right_free,
            enable_gap_close_penalty,
            rust_scaler.as_ref(),
        )?);
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    seq1,
    seq2,
    seq3,
    *,
    score_matrix,
    gap_open=-5.0,
    gap_extend=-1.0,
    seq1_left_free=false,
    seq1_right_free=false,
    seq2_left_free=false,
    seq2_right_free=false,
    seq3_left_free=false,
    seq3_right_free=false,
    enable_gap_close_penalty=true
))]
fn needleman_wunsch_3d(
    seq1: &str,
    seq2: &str,
    seq3: &str,
    score_matrix: &Bound<'_, PyDict>,
    gap_open: f64,
    gap_extend: f64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    seq3_left_free: bool,
    seq3_right_free: bool,
    enable_gap_close_penalty: bool,
) -> PyResult<(String, String, String, f64)> {
    let table = build_score_table(score_matrix)?;

    let a: Vec<u8> = seq1
        .as_bytes()
        .iter()
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let b: Vec<u8> = seq2
        .as_bytes()
        .iter()
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let c: Vec<u8> = seq3
        .as_bytes()
        .iter()
        .map(|ch| ch.to_ascii_uppercase())
        .collect();

    let n = a.len();
    let m = b.len();
    let l3 = c.len();

    let masks = [0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8];
    let step_i = [1_usize, 0, 1, 0, 1, 0, 1];
    let step_j = [1_usize, 1, 0, 0, 1, 1, 0];
    let step_k = [1_usize, 1, 1, 1, 0, 0, 0];
    let (effective_gap_open, gap_close_penalty) =
        resolve_gap_costs(gap_open, gap_extend, enable_gap_close_penalty);

    let cell = (n + 1) * (m + 1) * (l3 + 1);
    let idx = |s: usize, i: usize, j: usize, k: usize| -> usize {
        s * cell + ((i * (m + 1) + j) * (l3 + 1) + k)
    };

    let mut dp = vec![NEG_INF; 7 * cell];
    let mut ptr_state = vec![-1_i8; 7 * cell];
    let mut ptr_di = vec![0_i8; 7 * cell];
    let mut ptr_dj = vec![0_i8; 7 * cell];
    let mut ptr_dk = vec![0_i8; 7 * cell];

    dp[idx(0, 0, 0, 0)] = 0.0;
    ptr_state[idx(0, 0, 0, 0)] = 0;

    for i in 0..=n {
        for j in 0..=m {
            for k in 0..=l3 {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }

                for s in 0..7 {
                    let mask = masks[s];
                    let di = step_i[s];
                    let dj = step_j[s];
                    let dk = step_k[s];

                    if i < di || j < dj || k < dk {
                        continue;
                    }

                    let pi = i - di;
                    let pj = j - dj;
                    let pk = k - dk;

                    let mut letters = [0_u8; 3];
                    let mut cnt = 0_usize;
                    if (mask & 1) == 0 {
                        letters[cnt] = a[i - 1];
                        cnt += 1;
                    }
                    if (mask & 2) == 0 {
                        letters[cnt] = b[j - 1];
                        cnt += 1;
                    }
                    if (mask & 4) == 0 {
                        letters[cnt] = c[k - 1];
                        cnt += 1;
                    }

                    let mut sub = 0.0;
                    for x in 0..cnt {
                        for y in (x + 1)..cnt {
                            sub += table_score(&table, letters[x], letters[y]);
                        }
                    }

                    let mut best = NEG_INF;
                    let mut best_prev = -1_i8;

                    for ps in 0..7 {
                        let prev_mask = masks[ps];
                        let prev = dp[idx(ps, pi, pj, pk)];
                        if prev == NEG_INF {
                            continue;
                        }

                        let mut gap_pen = 0.0;

                        if (mask & 1) != 0 {
                            if !((i == 0 && seq1_left_free) || (i == n && seq1_right_free)) {
                                gap_pen += if (prev_mask & 1) != 0 {
                                    gap_extend
                                } else {
                                    effective_gap_open
                                };
                            }
                        } else if (prev_mask & 1) != 0 && !(seq1_left_free && i == 1) {
                            gap_pen += gap_close_penalty;
                        }

                        if (mask & 2) != 0 {
                            if !((j == 0 && seq2_left_free) || (j == m && seq2_right_free)) {
                                gap_pen += if (prev_mask & 2) != 0 {
                                    gap_extend
                                } else {
                                    effective_gap_open
                                };
                            }
                        } else if (prev_mask & 2) != 0 && !(seq2_left_free && j == 1) {
                            gap_pen += gap_close_penalty;
                        }

                        if (mask & 4) != 0 {
                            if !((k == 0 && seq3_left_free) || (k == l3 && seq3_right_free)) {
                                gap_pen += if (prev_mask & 4) != 0 {
                                    gap_extend
                                } else {
                                    effective_gap_open
                                };
                            }
                        } else if (prev_mask & 4) != 0 && !(seq3_left_free && k == 1) {
                            gap_pen += gap_close_penalty;
                        }

                        let cand = prev + sub + gap_pen;
                        if cand > best {
                            best = cand;
                            best_prev = ps as i8;
                        }
                    }

                    if best_prev >= 0 {
                        let at = idx(s, i, j, k);
                        dp[at] = best;
                        ptr_state[at] = best_prev;
                        ptr_di[at] = di as i8;
                        ptr_dj[at] = dj as i8;
                        ptr_dk[at] = dk as i8;
                    }
                }
            }
        }
    }

    let mut end_scores = [NEG_INF; 7];
    for s in 0..7 {
        let mut sc = dp[idx(s, n, m, l3)];
        if sc != NEG_INF && gap_close_penalty != 0.0 {
            let mask = masks[s];
            if (mask & 1) != 0 && !seq1_right_free {
                sc += gap_close_penalty;
            }
            if (mask & 2) != 0 && !seq2_right_free {
                sc += gap_close_penalty;
            }
            if (mask & 4) != 0 && !seq3_right_free {
                sc += gap_close_penalty;
            }
        }
        end_scores[s] = sc;
    }

    let mut best_state = 0_usize;
    let mut best_score = end_scores[0];
    for s in 1..7 {
        let sc = end_scores[s];
        if sc > best_score {
            best_score = sc;
            best_state = s;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut k = l3;
    let mut state = best_state;
    let mut out1: Vec<u8> = Vec::with_capacity(n + m + l3);
    let mut out2: Vec<u8> = Vec::with_capacity(n + m + l3);
    let mut out3: Vec<u8> = Vec::with_capacity(n + m + l3);

    while i > 0 || j > 0 || k > 0 {
        let mask = masks[state];
        if (mask & 1) != 0 {
            out1.push(b'-');
        } else {
            out1.push(a[i - 1]);
        }
        if (mask & 2) != 0 {
            out2.push(b'-');
        } else {
            out2.push(b[j - 1]);
        }
        if (mask & 4) != 0 {
            out3.push(b'-');
        } else {
            out3.push(c[k - 1]);
        }

        let at = idx(state, i, j, k);
        let prev_state = ptr_state[at];
        if prev_state < 0 {
            return Err(PyValueError::new_err(
                "unset pointer encountered during 3D traceback",
            ));
        }
        let di = ptr_di[at] as usize;
        let dj = ptr_dj[at] as usize;
        let dk = ptr_dk[at] as usize;
        i -= di;
        j -= dj;
        k -= dk;
        state = prev_state as usize;
    }

    out1.reverse();
    out2.reverse();
    out3.reverse();

    Ok((
        String::from_utf8(out1).unwrap_or_default(),
        String::from_utf8(out2).unwrap_or_default(),
        String::from_utf8(out3).unwrap_or_default(),
        best_score,
    ))
}

#[pymodule]
fn sgad_rust_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustScoreScaler>()?;
    m.add_function(wrap_pyfunction!(make_rust_score_scaler, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_batch, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_3d, m)?)?;
    Ok(())
}
