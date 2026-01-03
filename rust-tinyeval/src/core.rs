//! Core types and async API client for tinyeval

use crate::error::{Result, TinyEvalError};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::DynamicImage;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;

/// Represents an image in a prompt - either base64 encoded or raw image data
#[derive(Debug, Clone)]
pub enum PromptImage {
    Base64(String),
    Image(DynamicImage),
}

impl PromptImage {
    /// Encode the image to base64 data URL format
    pub fn to_data_url(&self) -> Result<String> {
        match self {
            PromptImage::Base64(s) => {
                if s.starts_with("data:") {
                    Ok(s.clone())
                } else {
                    Ok(format!("data:image/png;base64,{}", s))
                }
            }
            PromptImage::Image(img) => {
                let rgb_img = img.to_rgb8();
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                rgb_img.write_to(&mut cursor, image::ImageFormat::Png)?;
                let encoded = BASE64.encode(&buffer);
                Ok(format!("data:image/png;base64,{}", encoded))
            }
        }
    }

    /// Get bytes for hashing
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        match self {
            PromptImage::Base64(s) => {
                let data = if s.starts_with("data:") {
                    s.split(',').nth(1).unwrap_or(s)
                } else {
                    s
                };
                BASE64.decode(data).map_err(TinyEvalError::from)
            }
            PromptImage::Image(img) => {
                let rgb_img = img.to_rgb8();
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                rgb_img.write_to(&mut cursor, image::ImageFormat::Png)?;
                Ok(buffer)
            }
        }
    }
}

/// A single evaluation sample
#[derive(Debug, Clone)]
pub struct Sample {
    pub prompt: String,
    pub target: String,
    pub images: Vec<PromptImage>,
}

impl Sample {
    /// Create a text-only sample
    pub fn text(prompt: String, target: String) -> Self {
        Self {
            prompt,
            target,
            images: vec![],
        }
    }

    /// Create a multimodal sample with images
    pub fn multimodal(prompt: String, target: String, images: Vec<PromptImage>) -> Self {
        Self {
            prompt,
            target,
            images,
        }
    }
}

/// Task definition with sample loader and scoring function
pub struct Task {
    pub name: String,
    pub loader: Box<dyn Fn(Option<usize>, u64) -> Result<Vec<Sample>> + Send + Sync>,
    pub scorer: Box<dyn Fn(&str, &str) -> f64 + Send + Sync>,
}

impl Task {
    pub fn new<L, S>(name: &str, loader: L, scorer: S) -> Self
    where
        L: Fn(Option<usize>, u64) -> Result<Vec<Sample>> + Send + Sync + 'static,
        S: Fn(&str, &str) -> f64 + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            loader: Box::new(loader),
            scorer: Box::new(scorer),
        }
    }

    /// Load samples for this task
    pub fn load_samples(&self, max_samples: Option<usize>, seed: u64) -> Result<Vec<Sample>> {
        (self.loader)(max_samples, seed)
    }

    /// Score a response against a target
    pub fn score(&self, response: &str, target: &str) -> f64 {
        (self.scorer)(response, target)
    }
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIConfig {
    pub url: String,
    pub model: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_num_concurrent")]
    pub num_concurrent: usize,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    pub api_key: Option<String>,
}

fn default_seed() -> u64 {
    42
}
fn default_num_concurrent() -> usize {
    8
}
fn default_timeout() -> u64 {
    120
}
fn default_max_retries() -> u32 {
    3
}

impl APIConfig {
    pub fn new(url: String, model: String) -> Self {
        Self {
            url,
            model,
            seed: 42,
            num_concurrent: 8,
            timeout_seconds: 120,
            max_retries: 3,
            api_key: None,
        }
    }

    /// Parse from key=value format string
    pub fn from_model_args(args: &str) -> Result<Self> {
        let mut url = None;
        let mut model = None;
        let mut seed = 42u64;
        let mut num_concurrent = 8usize;
        let mut timeout = 120u64;
        let mut max_retries = 3u32;
        let mut api_key = None;

        for part in args.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| TinyEvalError::InvalidModelArgs(format!("Invalid format: {}", part)))?;

            match key.trim() {
                "base_url" => url = Some(value.trim().to_string()),
                "model" => model = Some(value.trim().to_string()),
                "seed" => {
                    seed = value
                        .trim()
                        .parse()
                        .map_err(|_| TinyEvalError::ParseError(format!("Invalid seed: {}", value)))?
                }
                "num_concurrent" => {
                    num_concurrent = value.trim().parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid num_concurrent: {}", value))
                    })?
                }
                "timeout" => {
                    timeout = value.trim().parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid timeout: {}", value))
                    })?
                }
                "max_retries" => {
                    max_retries = value.trim().parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid max_retries: {}", value))
                    })?
                }
                "api_key" => api_key = Some(value.trim().to_string()),
                _ => {} // Ignore unknown keys
            }
        }

        let url = url.ok_or_else(|| TinyEvalError::MissingField("base_url".to_string()))?;
        let model = model.ok_or_else(|| TinyEvalError::MissingField("model".to_string()))?;

        Ok(Self {
            url: format!("{}/chat/completions", url.trim_end_matches('/')),
            model,
            seed,
            num_concurrent,
            timeout_seconds: timeout,
            max_retries,
            api_key,
        })
    }
}

/// Metrics from an evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub exact_match: f64,
}

/// Per-sample result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggedSample {
    pub doc_id: usize,
    pub target: String,
    pub prompt: String,
    pub response: String,
    pub exact_match: f64,
}

/// Result from running a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task: String,
    pub task_hash: String,
    pub metrics: Metrics,
    pub num_samples: usize,
    pub elapsed: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<LoggedSample>,
}

/// Generation kwargs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenKwargs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl GenKwargs {
    /// Parse from key=value format string
    pub fn from_str(args: &str) -> Result<Self> {
        let mut kwargs = GenKwargs::default();

        for part in args.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| TinyEvalError::ParseError(format!("Invalid format: {}", part)))?;

            let key = key.trim();
            let value = value.trim();

            match key {
                "temperature" => {
                    kwargs.temperature = Some(value.parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid temperature: {}", value))
                    })?)
                }
                "max_tokens" => {
                    kwargs.max_tokens = Some(value.parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid max_tokens: {}", value))
                    })?)
                }
                "top_p" => {
                    kwargs.top_p = Some(value.parse().map_err(|_| {
                        TinyEvalError::ParseError(format!("Invalid top_p: {}", value))
                    })?)
                }
                _ => {
                    // Try to parse as JSON value
                    let json_value: serde_json::Value = serde_json::from_str(value)
                        .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
                    kwargs.extra.insert(key.to_string(), json_value);
                }
            }
        }

        Ok(kwargs)
    }
}

/// OpenAI chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatMessage {
    pub fn user(text: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::Value::String(text.to_string()),
        }
    }

    pub fn user_with_images(text: &str, images: &[PromptImage]) -> Result<Self> {
        let mut content = vec![serde_json::json!({
            "type": "text",
            "text": text
        })];

        for image in images {
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": image.to_data_url()?
                }
            }));
        }

        Ok(Self {
            role: "user".to_string(),
            content: serde_json::Value::Array(content),
        })
    }
}

/// OpenAI chat completion request
#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(flatten)]
    extra: std::collections::HashMap<String, serde_json::Value>,
}

/// OpenAI chat completion response
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// Async client for OpenAI-compatible APIs
pub struct ApiClient {
    client: Client,
    config: APIConfig,
    semaphore: Semaphore,
}

impl ApiClient {
    pub fn new(config: APIConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to build HTTP client");

        let semaphore = Semaphore::new(config.num_concurrent);

        Self {
            client,
            config,
            semaphore,
        }
    }

    /// Send a single chat completion request with retries
    async fn complete_one(
        &self,
        sample: &Sample,
        gen_kwargs: &GenKwargs,
    ) -> Result<String> {
        let message = if sample.images.is_empty() {
            ChatMessage::user(&sample.prompt)
        } else {
            ChatMessage::user_with_images(&sample.prompt, &sample.images)?
        };

        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![message],
            temperature: gen_kwargs.temperature,
            max_tokens: gen_kwargs.max_tokens,
            top_p: gen_kwargs.top_p,
            seed: Some(self.config.seed),
            extra: gen_kwargs.extra.clone(),
        };

        let mut last_error = None;
        let mut delay = Duration::from_secs(1);

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                sleep(delay).await;
                delay = std::cmp::min(delay * 2, Duration::from_secs(8));
            }

            let mut req = self.client.post(&self.config.url).json(&request);

            if let Some(ref api_key) = self.config.api_key {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }

            match req.send().await {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let body: ChatCompletionResponse = response.json().await?;
                        if let Some(choice) = body.choices.first() {
                            return Ok(choice.message.content.clone());
                        }
                        return Err(TinyEvalError::ApiError("No choices in response".to_string()));
                    }

                    if status.as_u16() == 429 {
                        last_error = Some(TinyEvalError::RateLimited(delay.as_secs()));
                        continue;
                    }

                    let error_text = response.text().await.unwrap_or_default();
                    return Err(TinyEvalError::ApiError(format!(
                        "HTTP {}: {}",
                        status, error_text
                    )));
                }
                Err(e) => {
                    if e.is_timeout() {
                        last_error = Some(TinyEvalError::Timeout(self.config.timeout_seconds));
                        continue;
                    }
                    last_error = Some(TinyEvalError::HttpError(e));
                }
            }
        }

        Err(TinyEvalError::MaxRetriesExceeded(
            self.config.max_retries,
            last_error.map(|e| e.to_string()).unwrap_or_default(),
        ))
    }

    /// Run completions for multiple samples concurrently
    pub async fn complete_batch(
        &self,
        samples: &[Sample],
        gen_kwargs: &GenKwargs,
    ) -> Vec<Result<String>> {
        let futures: Vec<_> = samples
            .iter()
            .map(|sample| async {
                let _permit = self.semaphore.acquire().await.unwrap();
                self.complete_one(sample, gen_kwargs).await
            })
            .collect();

        futures::future::join_all(futures).await
    }
}

/// Compute SHA256 hash of task samples for reproducibility
pub fn compute_task_hash(samples: &[Sample]) -> Result<String> {
    let mut hasher = Sha256::new();

    for sample in samples {
        hasher.update(sample.prompt.as_bytes());
        hasher.update(sample.target.as_bytes());

        for image in &sample.images {
            hasher.update(&image.to_bytes()?);
        }
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Normalize text for comparison (match Python implementation)
pub fn normalize(text: &str) -> String {
    text.replace('$', "")
        .replace(',', "")
        .replace('%', "")
        .replace("####", "")
        .replace('.', "")
        .trim()
        .to_lowercase()
}

/// Run a task and return results
pub async fn run_task(
    task: &Task,
    config: &APIConfig,
    gen_kwargs: &GenKwargs,
    max_samples: Option<usize>,
    log_samples: bool,
) -> Result<TaskResult> {
    let start = Instant::now();

    // Load samples
    let samples = task.load_samples(max_samples, config.seed)?;
    let num_samples = samples.len();

    // Compute task hash
    let task_hash = compute_task_hash(&samples)?;

    // Create API client and run completions
    let client = ApiClient::new(config.clone());
    let responses = client.complete_batch(&samples, gen_kwargs).await;

    // Score responses
    let mut total_score = 0.0;
    let mut logged_samples = Vec::new();

    for (i, (sample, response_result)) in samples.iter().zip(responses.iter()).enumerate() {
        let response = response_result
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("");

        let score = task.score(response, &sample.target);
        total_score += score;

        if log_samples {
            logged_samples.push(LoggedSample {
                doc_id: i,
                target: sample.target.clone(),
                prompt: sample.prompt.clone(),
                response: response.to_string(),
                exact_match: score,
            });
        }
    }

    let exact_match = if num_samples > 0 {
        total_score / num_samples as f64
    } else {
        0.0
    };

    Ok(TaskResult {
        task: task.name.clone(),
        task_hash,
        metrics: Metrics { exact_match },
        num_samples,
        elapsed: start.elapsed().as_secs_f64(),
        samples: logged_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("$1,234.56"), "123456");
        assert_eq!(normalize("50%"), "50");
        assert_eq!(normalize("#### 42"), "42");
        assert_eq!(normalize("Hello World"), "hello world");
    }

    #[test]
    fn test_api_config_from_model_args() {
        let config = APIConfig::from_model_args(
            "model=gpt-4,base_url=http://localhost:8000/v1,seed=123",
        )
        .unwrap();

        assert_eq!(config.model, "gpt-4");
        assert_eq!(
            config.url,
            "http://localhost:8000/v1/chat/completions"
        );
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_gen_kwargs_from_str() {
        let kwargs = GenKwargs::from_str("temperature=0.7,max_tokens=100").unwrap();
        assert_eq!(kwargs.temperature, Some(0.7));
        assert_eq!(kwargs.max_tokens, Some(100));
    }

    #[test]
    fn test_compute_task_hash_deterministic() {
        let samples = vec![
            Sample::text("What is 2+2?".to_string(), "4".to_string()),
            Sample::text("What is 3+3?".to_string(), "6".to_string()),
        ];

        let hash1 = compute_task_hash(&samples).unwrap();
        let hash2 = compute_task_hash(&samples).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_task_hash_changes_with_content() {
        let samples1 = vec![Sample::text("What is 2+2?".to_string(), "4".to_string())];
        let samples2 = vec![Sample::text("What is 2+3?".to_string(), "5".to_string())];

        let hash1 = compute_task_hash(&samples1).unwrap();
        let hash2 = compute_task_hash(&samples2).unwrap();

        assert_ne!(hash1, hash2);
    }
}
