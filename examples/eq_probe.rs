//! One-shot wall-clock probe for try_eq_by (M13 validation): times each
//! case ONCE with Instant, no criterion loop, inputs laundered through
//! std::hint::black_box so nothing can be hoisted or folded. Run:
//! cargo run --release --example eq_probe
use champ::PVector;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    const N: i64 = 240_000;
    let items: Vec<i64> = (0..N).collect();
    let base = PVector::from_slice(&items);
    let edited = base.clone().set_owned((N - 1) as usize, -1);
    let disjoint = PVector::from_slice(&items);

    for (name, a, b) in [
        ("identical", &base, &base.clone()),
        ("shared_one_edit", &base, &edited),
        ("disjoint_equal", &base, &disjoint),
    ] {
        // warm caches once, then measure 100 runs and report the mean.
        let _ = black_box(black_box(a).eq_by(black_box(b), |x, y| x == y));
        let t = Instant::now();
        let mut r1 = false;
        for _ in 0..100 {
            r1 = black_box(black_box(a).eq_by(black_box(b), |x, y| x == y));
        }
        let per = t.elapsed().as_nanos() / 100;
        let t2 = Instant::now();
        let mut r2 = false;
        for _ in 0..100 {
            r2 = black_box(black_box(a) == black_box(b));
        }
        let per2 = t2.elapsed().as_nanos() / 100;
        println!("{name}: eq_by {per} ns (result {r1}), partial_eq {per2} ns (result {r2})");
    }
}
