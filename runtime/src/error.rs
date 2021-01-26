use crate::{Allocator, Error, InvalidKind, Invalidated};

use std::ops::Range;

fn span_to_string(span: &Range<usize>, line_offsets: &[usize]) -> String {
    let line_start = match line_offsets.binary_search(&span.start) {
        Ok(x) => x,
        Err(x) => x - 1,
    };
    let line_end = match line_offsets.binary_search(&span.end) {
        Ok(x) => x,
        Err(x) => x - 1,
    };

    let col_start = span.start - line_offsets[line_start];
    let col_end = span.end - line_offsets[line_end];

    format!(
        "span({}:{}..{}:{})",
        1 + line_start,
        1 + col_start,
        1 + line_end,
        1 + col_end
    )
}

struct DisplayToDebug<T>(T);

use std::fmt;
impl<T: fmt::Display> fmt::Debug for DisplayToDebug<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
}

impl<T: fmt::Display> fmt::Display for DisplayToDebug<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
}

impl<T: fmt::Display> std::error::Error for DisplayToDebug<T> {}

#[cold]
#[inline(never)]
pub(crate) fn handle_error(
    err: Error,
    span: std::ops::Range<usize>,
    allocator: &Allocator,
    line_offsets: &[usize],
) -> Box<dyn std::error::Error> {
    use memory_model::alias::Error::*;

    let span = span_to_string(&span, line_offsets);

    let err = match err {
        Error::Alias(err) => err,
        Error::InvalidPtr(ptr) => {
            let err = match allocator.invalidated.get(ptr) {
                Some(Invalidated {
                    kind: InvalidKind::Freed,
                    span: freed_span,
                }) => format!(
                    "{}: Use of freed pointer `{ptr}`. Note: freed `{ptr}` at {freed_span}",
                    span,
                    ptr = ptr,
                    freed_span = span_to_string(&freed_span, line_offsets)
                ),
                Some(Invalidated {
                    kind: InvalidKind::Moved,
                    span: moved_span,
                }) => format!(
                    "{}: Use of moved pointer `{ptr}`. Note: moved `{ptr}` at {moved_span}",
                    span,
                    ptr = ptr,
                    moved_span = span_to_string(&moved_span, line_offsets)
                ),
                None => format!("{}: Unknown pointer `{}`", span, ptr),
            };

            return DisplayToDebug(err).into()
        }

        Error::NoFunction { func } => return DisplayToDebug(format!("{}: Unkown function {}", span, func)).into(),
        Error::TypeMismatch { arg, farg } => {
            let read = ["", " read"];
            let write = ["", " write"];
            let modifier = ["shr", "exc"];

            let arg_ty = arg.ty.unwrap();
            let farg_ty = farg.ty.unwrap();

            match (
                format_args!(
                    "@{}{}{}",
                    modifier[usize::from(arg_ty.is_exclusive)],
                    read[usize::from(arg_ty.read)],
                    write[usize::from(arg_ty.write)]
                ),
                format_args!(
                    "@{}{}{}",
                    modifier[usize::from(farg_ty.is_exclusive)],
                    read[usize::from(farg_ty.read)],
                    write[usize::from(farg_ty.write)]
                ),
            ) {
                (arg_ty, farg_ty) => {
                    return DisplayToDebug(format!(
                        "\
{}: Type mismatch: argument `{}` (at {}) didn't match declared argument `{}` (at {}):
expected `{}`, but got `{}`",
                        span,
                        arg.name,
                        span_to_string(&arg.span, line_offsets),
                        farg.name,
                        span_to_string(&farg.span, line_offsets),
                        farg_ty,
                        arg_ty,
                    ))
                    .into()
                }
            }
        }
    };

    let err = match err {
        ReborrowInvalidatesSource { ptr, source } => format!(
            "{}: Could not borrow `{}`, because it invalidated it's source `{}`",
            span,
            allocator.name(ptr),
            allocator.name(source)
        ),
        UseAfterFree(ptr) => format!("{}: Tried to use `{}` after it was freed", span, allocator.name(ptr)),
        InvalidPtr(ptr) => format!(
            "{}: Tried to use `{}`, which was never registered",
            span,
            allocator.name(ptr)
        ),
        NotExclusive(ptr) => format!(
            "{}: Tried to use `{}` exclusively, but it is shared",
            span,
            allocator.name(ptr)
        ),
        NotShared(ptr) => format!(
            "{}: Tried to use `{}` as shared, but it is exclusive",
            span,
            allocator.name(ptr)
        ),
        DeallocateNonOwning(ptr) => format!(
            "{}: Tried to deallocate `{}`, but it doesn't own an allocation",
            span,
            allocator.name(ptr)
        ),
        InvalidatesOldMeta(ptr) => format!(
            "{}: Tried to update the meta data of `{}`, but it would invalidate itself",
            span,
            allocator.name(ptr)
        ),
        AllocateRangeOccupied { ptr: _, range } => format!(
            "{}: Tried to allocate in range {:?}, but that range is already occupied",
            span, range
        ),
        InvalidForRange { ptr, range } => format!(
            "{}: Tried to use `{}` for the range {:?}, but it is not valid for that range",
            span,
            allocator.name(ptr),
            range
        ),
        ReborrowSubset {
            ptr,
            source,
            source_range,
        } => format!(
            "{}: Tried to reborrow `{}` from `{2}` for the range {3:?}, but `{2}` is not valid for that range",
            span,
            allocator.name(ptr),
            allocator.name(source),
            source_range
        ),
    };

    DisplayToDebug(err).into()
}
