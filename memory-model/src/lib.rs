#![forbid(unsafe_code)]

pub mod alias;

mod recycle;

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pointer(std::num::NonZeroU32);

impl Pointer {
    pub fn create() -> Self {
        use std::{
            num::NonZeroU32,
            sync::atomic::{AtomicU32, Ordering::Relaxed},
        };

        static NEXT_PTR_ID: AtomicU32 = AtomicU32::new(0);

        let id = NEXT_PTR_ID
            .fetch_add(1, Relaxed)
            .checked_add(1)
            .expect("Tried to create too many pointers");

        let id = NonZeroU32::new(id).unwrap();

        Pointer(id)
    }
}

impl fmt::Debug for Pointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ptr({})", self.0) }
}
