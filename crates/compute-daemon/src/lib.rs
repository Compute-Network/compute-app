#![allow(
    clippy::cloned_ref_to_slice_refs,
    clippy::collapsible_if,
    clippy::derivable_impls,
    clippy::format_in_format_args,
    clippy::if_same_then_else,
    clippy::large_enum_variant,
    clippy::len_without_is_empty,
    clippy::let_and_return,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::manual_strip,
    clippy::map_entry,
    clippy::needless_as_bytes,
    clippy::needless_borrow,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::useless_conversion,
    dead_code
)]

pub mod benchmark;
pub mod config;
pub mod container;
pub mod daemon;
pub mod error;
pub mod hardware;
pub mod idle;
pub mod inference;
pub mod logging;
pub mod metrics;
pub mod prototype_chain;
pub mod real_chain;
pub mod relay;
pub mod runtime;
pub mod service;
pub mod stage_runtime;
