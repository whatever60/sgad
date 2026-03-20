use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sgad_rust_native::bench_needleman_wunsch_2d_with_scaler;

/// Build a dense ASCII substitution table used by the Rust NW core.
fn make_score_table() -> Vec<f64> {
    let mut table = vec![-1.0_f64; 256 * 256];
    for &base in b"ACGT" {
        let idx = (base as usize) * 256 + (base as usize);
        table[idx] = 1.0;
    }
    table
}

/// Benchmark short-sequence 2D Needleman-Wunsch runtime for scaler-enabled mode.
fn bench_nw2d_short(c: &mut Criterion) {
    let table = make_score_table();
    let mut group = c.benchmark_group("nw2d_short_with_scaler");

    let cases = [
        ("len16", "ACGTTGCAACGTACGT", "ACGTCGCAACGTTCGT"),
        (
            "len28",
            "ACGTTGCAACGTACGTACGTTGCAACGT",
            "ACGTCGCAACGTTCGTACGTCGCAACGT",
        ),
    ];

    for (name, seq1, seq2) in cases {
        group.bench_with_input(BenchmarkId::new("pair", name), &(seq1, seq2), |b, input| {
            b.iter(|| {
                let (lhs, rhs, score) = bench_needleman_wunsch_2d_with_scaler(
                    black_box(input.0),
                    black_box(input.1),
                    black_box(&table),
                    black_box(-5.0),
                    black_box(-1.0),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                    black_box(1.0),
                    black_box(1.0),
                );
                black_box(lhs);
                black_box(rhs);
                black_box(score);
            });
        });
    }

    group.finish();
}

criterion_group!(nw2d_short, bench_nw2d_short);
criterion_main!(nw2d_short);
