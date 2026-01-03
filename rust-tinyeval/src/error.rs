//! Error types for tinyeval

use thiserror::Error;

/// Main error type for tinyeval
#[derive(Error, Debug)]
pub enum TinyEvalError {
    #[error("Unknown task: {0}. Available tasks: {1}")]
    UnknownTask(String, String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    #[error("Invalid model args: {0}")]
    InvalidModelArgs(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Task loading error: {0}")]
    TaskLoadError(String),

    #[error("Rate limited by API, retry after {0} seconds")]
    RateLimited(u64),

    #[error("Request timeout after {0} seconds")]
    Timeout(u64),

    #[error("Max retries ({0}) exceeded: {1}")]
    MaxRetriesExceeded(u32, String),

    #[error("Unsupported image type: {0}")]
    UnsupportedImageType(String),

    #[error("Dataset error: {0}")]
    DatasetError(String),
}

/// Result type alias for tinyeval
pub type Result<T> = std::result::Result<T, TinyEvalError>;
