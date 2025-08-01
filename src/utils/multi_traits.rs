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
