use champ::{PText, PVector, PersistentHashMap};

#[test]
fn readme_map_snippet() {
    let empty: PersistentHashMap<&str, i32> = PersistentHashMap::new();

    let a = empty.assoc("x", 1).assoc("y", 2);
    let b = a.assoc("x", 10);
    assert_eq!(a.get(&"x"), Some(&1));
    assert_eq!(b.get(&"x"), Some(&10));

    let big = (0..100_000).fold(PersistentHashMap::new(), |m, i| m.assoc_owned(i, i * 2));

    let mut t = big.transient();
    t.assoc(100_000, 0);
    t.dissoc(&0);
    let done = t.persistent();
    assert_eq!(done.len(), 100_000);
}

#[test]
fn readme_vector_snippet() {
    let v = PVector::from_slice(&[1, 2, 3]);
    let w = v.push_back(4);
    let x = w.clone().set_owned(0, 99);

    assert_eq!(v.len(), 3);
    assert_eq!(x.get(0), Some(&99));

    let front = v.slice(0..2);
    assert_eq!(front.len(), 2);
    let sum: i32 = w.iter().sum();
    assert_eq!(sum, 10);

    let equal = w.eq_by(&w.clone(), |a, b| a == b);
    assert!(equal);
}

#[test]
fn readme_text_snippet() {
    let t = PText::from("hello world");
    let u = t.splice(5..6, ", ");
    assert_eq!(u.to_string(), "hello, world");
    assert_eq!(t.to_string(), "hello world");

    let head = u.slice(0..5);
    assert_eq!(head.to_string(), "hello");
    let line = u.char_to_line(3);
    assert_eq!(line, 0);
    let joined: String = u.chunks().collect();
    assert_eq!(joined, "hello, world");
}
