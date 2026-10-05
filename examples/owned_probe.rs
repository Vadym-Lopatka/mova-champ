//! One-shot Instant probe for the owned-push build (SPEC-M11), same
//! pattern as `eq_probe.rs`: black_boxed inputs, discarded warmups,
//! min-of-rounds printed for external interleaved A/B comparison.

use std::hint::black_box;
use std::time::Instant;

fn build(items: &[i64]) -> champ::PVector<i64> {
    let mut v = champ::PVector::new();
    for &x in black_box(items) {
        v = v.push_back_owned(x);
    }
    v
}

fn build_imbl(items: &[i64]) -> imbl::Vector<i64> {
    let mut v = imbl::Vector::new();
    for &x in black_box(items) {
        v.push_back(x);
    }
    v
}

fn main() {
    let items: Vec<i64> = (0..240_000).collect();
    for _ in 0..3 {
        black_box(build(&items).len());
        black_box(build_imbl(&items).len());
    }
    let (mut best_pv, mut best_im) = (f64::MAX, f64::MAX);
    // Interleaved rounds: drop happens OUTSIDE the timed window for both.
    for _ in 0..10 {
        let t = Instant::now();
        let v = build(&items);
        let dt = t.elapsed().as_secs_f64();
        black_box(v.len());
        drop(v);
        if dt < best_pv {
            best_pv = dt;
        }
        let t = Instant::now();
        let v = build_imbl(&items);
        let dt = t.elapsed().as_secs_f64();
        black_box(v.len());
        drop(v);
        if dt < best_im {
            best_im = dt;
        }
    }
    println!("pvector {:.1} imbl {:.1} ratio {:.2}", best_pv * 1e6, best_im * 1e6, best_pv / best_im);
}
