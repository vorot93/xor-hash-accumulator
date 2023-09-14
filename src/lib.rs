use digest::{Digest, Output, OutputSizeUser};

pub struct Diff<V> {
    pub before: V,
    pub after: V,
}

pub trait Hashable {
    type HashableBytes<'a>: AsRef<[u8]>
    where
        Self: 'a;

    fn hashable_bytes(&self) -> Self::HashableBytes<'_>;
}

fn bitxor<T>(lhs: &mut Output<T>, rhs: Output<T>)
where
    T: OutputSizeUser,
{
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs ^= rhs;
    }
}

/// XOR-based accumulator for various data.
pub fn accumulate<D, V>(mut accumulator: Output<D>, diff: &Diff<V>) -> Output<D>
where
    D: Digest,
    V: Hashable,
{
    bitxor::<D>(
        &mut accumulator,
        D::digest(diff.before.hashable_bytes().as_ref()),
    );
    bitxor::<D>(
        &mut accumulator,
        D::digest(diff.after.hashable_bytes().as_ref()),
    );

    accumulator
}

#[cfg(feature = "rayon")]
pub fn accumulate_many<D, V>(mut accumulator: Output<D>, old_and_new_values: &[V]) -> Output<D>
where
    D: Digest,
    V: Hashable + Send + Sync,
{
    use rayon::prelude::*;

    bitxor::<D>(
        &mut accumulator,
        old_and_new_values
            .par_iter()
            .map(|value| D::digest(value.hashable_bytes().as_ref()))
            .reduce(Output::<D>::default, |mut lhs, rhs| {
                bitxor::<D>(&mut lhs, rhs);
                lhs
            }),
    );

    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_hash() {
        #[derive(Clone, Copy)]
        struct V(u8);

        impl Hashable for V {
            type HashableBytes<'a> = [u8; 1];

            fn hashable_bytes(&self) -> Self::HashableBytes<'_> {
                self.0.to_be_bytes()
            }
        }

        let starting_accumulator = [0; 32].into();

        let diff_incremental = [
            Diff {
                before: V(0),
                after: V(1),
            },
            Diff {
                before: V(1),
                after: V(2),
            },
            Diff {
                before: V(2),
                after: V(3),
            },
            Diff {
                before: V(3),
                after: V(4),
            },
        ];

        let diff_squashed = Diff {
            before: V(0),
            after: V(4),
        };

        let mut accumulator_incremental = starting_accumulator;
        for diff in &diff_incremental {
            accumulator_incremental =
                accumulate::<blake3::Hasher, _>(accumulator_incremental, diff);
        }
        let accumulator_squashed =
            accumulate::<blake3::Hasher, _>(starting_accumulator, &diff_squashed);

        assert_ne!(accumulator_incremental, starting_accumulator);
        assert_eq!(accumulator_incremental, accumulator_squashed);

        #[cfg(feature = "rayon")]
        {
            let mut values_parallel = vec![];
            for diff in &diff_incremental {
                values_parallel.push(diff.before);
                values_parallel.push(diff.after);
            }
            let accumulator_parallel =
                accumulate_many::<blake3::Hasher, _>(starting_accumulator, &values_parallel);
            assert_eq!(accumulator_incremental, accumulator_parallel);
        }
    }
}
