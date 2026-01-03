//! tinyeval - A minimal harness for evaluating LLMs through OpenAI-compatible APIs
//!
//! This crate provides:
//! - Core types for evaluation (Sample, Task, APIConfig, etc.)
//! - Async HTTP client for OpenAI-compatible APIs
//! - Task implementations (GSM8K, ChartQA)
//! - SHA256 hashing for reproducibility

pub mod core;
pub mod error;
pub mod tasks;

pub use crate::core::{
    compute_task_hash, normalize, run_task, APIConfig, ApiClient, ChatMessage, GenKwargs,
    LoggedSample, Metrics, PromptImage, Sample, Task, TaskResult,
};
pub use crate::error::{Result, TinyEvalError};
pub use crate::tasks::{available_tasks, get_task};
