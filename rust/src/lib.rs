use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

const NEG_INF: f64 = f64::NEG_INFINITY;

fn extract_ascii_base(key: &Bound<'_, PyAny>) -> PyResult<u8> {
    let s: String = key.extract()?;
    let bytes = s.as_bytes();
    if bytes.len() != 1 {
        return Err(PyValueError::new_err("score_matrix keys must be single characters"));
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

#[inline]
fn column_score_scale_factor(
    mask: u8,
    i: usize,
    j: usize,
    n: usize,
    m: usize,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    rust_scaler: Option<&RustScoreScaler>,
) -> PyResult<f64> {
    let (s1l, s1r, s2l, s2r) = match mask {
        0 => (
            i as isize - 1,
            n as isize - i as isize,
            j as isize - 1,
            m as isize - j as isize,
        ),
        1 => (
            i as isize,
            n as isize - i as isize,
            j as isize - 1,
            m as isize - j as isize,
        ),
        2 => (
            i as isize - 1,
            n as isize - i as isize,
            j as isize,
            m as isize - j as isize,
        ),
        _ => return Err(PyValueError::new_err("unsupported mask")),
    };

    if let Some(scaler) = rust_scaler {
        return Ok(scaler.factor(
            s1l,
            s1r,
            s2l,
            s2r,
            seq1_left_free,
            seq1_right_free,
            seq2_left_free,
            seq2_right_free,
        ));
    }

    // If no scaler object is supplied, use no scaling by default.
    Ok(1.0)
}

#[pyfunction]
#[pyo3(signature = (
    seq1,
    seq2,
    *,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=false,
    seq1_right_free=false,
    seq2_left_free=false,
    seq2_right_free=false,
    score_scaler_fn=None
))]
fn needleman_wunsch(
    py: Python<'_>,
    seq1: &str,
    seq2: &str,
    score_matrix: &Bound<'_, PyDict>,
    gap_open: i64,
    gap_extend: i64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    score_scaler_fn: Option<Py<RustScoreScaler>>,
) -> PyResult<(String, String, f64)> {
    let table = build_score_table(score_matrix)?;
    let rust_scaler = score_scaler_fn.map(|s| s.borrow(py).clone());

    let a: Vec<u8> = seq1.as_bytes().iter().map(|c| c.to_ascii_uppercase()).collect();
    let b: Vec<u8> = seq2.as_bytes().iter().map(|c| c.to_ascii_uppercase()).collect();
    let n = a.len();
    let m = b.len();

    let masks = [0_u8, 1_u8, 2_u8];
    let steps = [(1_usize, 1_usize), (0_usize, 1_usize), (1_usize, 0_usize)];

    let nm = (n + 1) * (m + 1);
    let idx = |s: usize, i: usize, j: usize| -> usize { s * nm + i * (m + 1) + j };

    let mut dp = vec![NEG_INF; 3 * nm];
    let mut ptr_state = vec![-1_i8; 3 * nm];
    let mut ptr_di = vec![0_i8; 3 * nm];
    let mut ptr_dj = vec![0_i8; 3 * nm];

    dp[idx(0, 0, 0)] = 0.0;
    ptr_state[idx(0, 0, 0)] = 0;

    for i in 0..=n {
        for j in 0..=m {
            if i == 0 && j == 0 {
                continue;
            }

            for (s, &mask) in masks.iter().enumerate() {
                let (di, dj) = steps[s];
                if i < di || j < dj {
                    continue;
                }

                let pi = i - di;
                let pj = j - dj;

                let sub = if mask == 0 {
                    let factor = column_score_scale_factor(
                        mask,
                        i,
                        j,
                        n,
                        m,
                        seq1_left_free,
                        seq1_right_free,
                        seq2_left_free,
                        seq2_right_free,
                        rust_scaler.as_ref(),
                    )?;
                    table_score(&table, a[i - 1], b[j - 1]) * factor
                } else {
                    0.0
                };

                let mut best = NEG_INF;
                let mut best_prev = -1_i8;

                for (ps, &prev_mask) in masks.iter().enumerate() {
                    let prev = dp[idx(ps, pi, pj)];
                    if prev == NEG_INF {
                        continue;
                    }

                    let factor = column_score_scale_factor(
                        mask,
                        i,
                        j,
                        n,
                        m,
                        seq1_left_free,
                        seq1_right_free,
                        seq2_left_free,
                        seq2_right_free,
                        rust_scaler.as_ref(),
                    )?;

                    let mut gap_pen = 0.0;
                    if (mask & 1) != 0 {
                        if !((i == 0 && seq1_left_free) || (i == n && seq1_right_free)) {
                            let base = if (prev_mask & 1) != 0 {
                                gap_extend as f64
                            } else {
                                gap_open as f64
                            };
                            gap_pen += base * factor;
                        }
                    }
                    if (mask & 2) != 0 {
                        if !((j == 0 && seq2_left_free) || (j == m && seq2_right_free)) {
                            let base = if (prev_mask & 2) != 0 {
                                gap_extend as f64
                            } else {
                                gap_open as f64
                            };
                            gap_pen += base * factor;
                        }
                    }

                    let cand = prev + sub + gap_pen;
                    if cand > best {
                        best = cand;
                        best_prev = ps as i8;
                    }
                }

                if best_prev >= 0 {
                    let at = idx(s, i, j);
                    dp[at] = best;
                    ptr_state[at] = best_prev;
                    ptr_di[at] = di as i8;
                    ptr_dj[at] = dj as i8;
                }
            }
        }
    }

    let mut best_state = 0_usize;
    let mut best_score = dp[idx(0, n, m)];
    for s in 1..3 {
        let sc = dp[idx(s, n, m)];
        if sc > best_score {
            best_score = sc;
            best_state = s;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut state = best_state;
    let mut out_a: Vec<u8> = Vec::with_capacity(n + m);
    let mut out_b: Vec<u8> = Vec::with_capacity(n + m);

    while i > 0 || j > 0 {
        let mask = masks[state];
        if (mask & 1) != 0 {
            out_a.push(b'-');
        } else {
            out_a.push(a[i - 1]);
        }
        if (mask & 2) != 0 {
            out_b.push(b'-');
        } else {
            out_b.push(b[j - 1]);
        }

        let at = idx(state, i, j);
        let prev_state = ptr_state[at];
        if prev_state < 0 {
            return Err(PyValueError::new_err("unset pointer encountered during traceback"));
        }
        let di = ptr_di[at] as usize;
        let dj = ptr_dj[at] as usize;
        i -= di;
        j -= dj;
        state = prev_state as usize;
    }

    out_a.reverse();
    out_b.reverse();

    Ok((
        String::from_utf8(out_a).unwrap_or_default(),
        String::from_utf8(out_b).unwrap_or_default(),
        best_score,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    seq1,
    seq2,
    seq3,
    *,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=false,
    seq1_right_free=false,
    seq2_left_free=false,
    seq2_right_free=false,
    seq3_left_free=false,
    seq3_right_free=false
))]
fn needleman_wunsch_3d(
    seq1: &str,
    seq2: &str,
    seq3: &str,
    score_matrix: &Bound<'_, PyDict>,
    gap_open: i64,
    gap_extend: i64,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    seq3_left_free: bool,
    seq3_right_free: bool,
) -> PyResult<(String, String, String, f64)> {
    let table = build_score_table(score_matrix)?;

    let a: Vec<u8> = seq1.as_bytes().iter().map(|c| c.to_ascii_uppercase()).collect();
    let b: Vec<u8> = seq2.as_bytes().iter().map(|c| c.to_ascii_uppercase()).collect();
    let c: Vec<u8> = seq3.as_bytes().iter().map(|ch| ch.to_ascii_uppercase()).collect();

    let n = a.len();
    let m = b.len();
    let l3 = c.len();

    let masks = [0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8];
    let step_i = [1_usize, 0, 1, 0, 1, 0, 1];
    let step_j = [1_usize, 1, 0, 0, 1, 1, 0];
    let step_k = [1_usize, 1, 1, 1, 0, 0, 0];

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
                                    gap_extend as f64
                                } else {
                                    gap_open as f64
                                };
                            }
                        }

                        if (mask & 2) != 0 {
                            if !((j == 0 && seq2_left_free) || (j == m && seq2_right_free)) {
                                gap_pen += if (prev_mask & 2) != 0 {
                                    gap_extend as f64
                                } else {
                                    gap_open as f64
                                };
                            }
                        }

                        if (mask & 4) != 0 {
                            if !((k == 0 && seq3_left_free) || (k == l3 && seq3_right_free)) {
                                gap_pen += if (prev_mask & 4) != 0 {
                                    gap_extend as f64
                                } else {
                                    gap_open as f64
                                };
                            }
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

    let mut best_state = 0_usize;
    let mut best_score = dp[idx(0, n, m, l3)];
    for s in 1..7 {
        let sc = dp[idx(s, n, m, l3)];
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
    m.add_function(wrap_pyfunction!(needleman_wunsch_3d, m)?)?;
    Ok(())
}
