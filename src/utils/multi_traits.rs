/*!
# Trait Enums

Sometimes, we want to return something from a function that implements trait `T` (often `Iterator`).
Often, however early returns or edge cases are not easily doable because the underlying struct is different.

Hence, in those cases we wrap all returns in an enum which only function it is to implement `T` using its inner values,
thus bypassing this problem.

If possibly however, try to keep such enums minimal as for bigger enums it is not clear that `rustc` can optimize the
`match self` away.
*/

/// Generates enums with the single purpose of allowing returns of different structs from a
/// function that only requires its return value to implement `Iterator<Item = I>`
macro_rules! impl_multi_iterators {
    ($(
        $name:ident -> $($T:ident:$G:ident),+;
    )*) => {
        $(
            pub enum $name<IterItem, $($G),+>
            where
                $(
                    $G: Iterator<Item = IterItem>,
                )+
            {
                $(
                    $T($G),
                )+
            }

            impl<IterItem, $($G),+> Iterator for $name<IterItem, $($G),+>
            where
                $(
                    $G: Iterator<Item = IterItem>,
                )+
            {
                type Item = IterItem;
                fn next(&mut self) -> Option<Self::Item> {
                    match self {
                        $(
                            $name::$T(iter) => iter.next(),
                        )+
                    }
                }
            }
        )*
    };
}

impl_multi_iterators!(
    DoubleIter -> IterA:A, IterB:B;
    TripleIter -> IterA:A, IterB:B, IterC:C;
);
