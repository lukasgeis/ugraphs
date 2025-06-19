/// A generalization over basic Set-Functionality
pub trait Set<T> {
    /// Inserts an element into the Set
    /// Returns *true* if the element was contained in the Set before
    fn insert(&mut self, value: T) -> bool;

    /// Inserts multiple elements into the Set
    fn insert_multiple<I: Iterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }

    /// Removes an element from the Set
    /// Returns *true* if the element was contained in the Set before
    fn remove(&mut self, value: T) -> bool;

    /// Removes multiple elements from the Set
    fn remove_multiple<I: Iterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.remove(value);
        }
    }

    /// Returns *true* if the element is contained in the Set
    fn contains(&self, value: T) -> bool;
}
