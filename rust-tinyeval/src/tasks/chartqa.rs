//! ChartQA evaluation - multimodal chart understanding

use crate::core::{PromptImage, Sample, Task};
use crate::error::{Result, TinyEvalError};
use once_cell::sync::Lazy;
use regex::Regex;

/// Regex for extracting "FINAL ANSWER:" format
static FINAL_ANSWER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)FINAL ANSWER:\s*(.+?)(?:\n|$)").unwrap()
});

/// Regex for cleaning numeric strings (remove $, %, and ,)
static NUMERIC_CLEAN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[$,%]").unwrap()
});

/// Format ChartQA prompt
fn format_chartqa_prompt(query: &str) -> String {
    format!(
        "<image>You are provided a chart image and will be asked a question. \
You have to think through your answer and provide a step-by-step solution. \
Once you have the solution, write the final answer in at most a few words at the end with the phrase \"FINAL ANSWER:\". \
The question is: {}\n\
Let's think step by step.",
        query
    )
}

/// ChartQA metric: exact match or 5% numeric tolerance
fn relaxed_match(response: &str, target: &str) -> f64 {
    let pred = if let Some(caps) = FINAL_ANSWER_RE.captures(response) {
        caps.get(1).map(|m| m.as_str().trim()).unwrap_or(response.trim())
    } else {
        response.trim()
    };

    // Exact match (case-insensitive)
    if pred.to_lowercase() == target.to_lowercase() {
        return 1.0;
    }

    // Try numeric comparison with 5% tolerance
    let pred_clean = NUMERIC_CLEAN_RE.replace_all(pred, "");
    let target_clean = NUMERIC_CLEAN_RE.replace_all(target, "");

    if let (Ok(pred_n), Ok(target_n)) = (
        pred_clean.parse::<f64>(),
        target_clean.parse::<f64>(),
    ) {
        if target_n == 0.0 {
            return if pred_n == 0.0 { 1.0 } else { 0.0 };
        }
        if (pred_n - target_n).abs() / target_n.abs() <= 0.05 {
            return 1.0;
        }
    }

    0.0
}

/// Sample ChartQA test questions (embedded for standalone operation)
/// Note: In production, these would be loaded from HuggingFace datasets with actual images
const CHARTQA_TEST_SAMPLES: &[(&str, &str)] = &[
    ("What is the value shown for 2020?", "45"),
    ("How much did sales increase from Q1 to Q2?", "25%"),
    ("What is the total revenue shown in the chart?", "$1,234"),
    ("Which category has the highest value?", "Technology"),
    ("What percentage does the blue segment represent?", "35"),
];

/// Load ChartQA samples
/// Note: Since we can't easily load images from HuggingFace in Rust,
/// this creates text-only samples for testing. Production use would
/// require a data loading mechanism for the actual dataset.
pub fn samples(max_samples: Option<usize>, _seed: u64) -> Result<Vec<Sample>> {
    let limit = max_samples.unwrap_or(CHARTQA_TEST_SAMPLES.len());

    // For testing, we create placeholder samples without actual images
    // In production, you would load actual ChartQA data with images
    let samples: Vec<Sample> = CHARTQA_TEST_SAMPLES
        .iter()
        .take(limit)
        .map(|(query, target)| {
            // Create a small placeholder image for testing
            let placeholder = create_placeholder_image();
            Sample::multimodal(
                format_chartqa_prompt(query),
                target.to_string(),
                vec![PromptImage::Image(placeholder)],
            )
        })
        .collect();

    if samples.is_empty() {
        return Err(TinyEvalError::TaskLoadError(
            "No ChartQA samples available".to_string(),
        ));
    }

    Ok(samples)
}

/// Create a small placeholder image for testing
fn create_placeholder_image() -> image::DynamicImage {
    use image::{Rgb, RgbImage};

    let mut img = RgbImage::new(100, 100);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([
            ((x * 2) % 256) as u8,
            ((y * 2) % 256) as u8,
            128,
        ]);
    }
    image::DynamicImage::ImageRgb8(img)
}

/// Score ChartQA response with relaxed matching (5% numeric tolerance)
pub fn score(response: &str, target: &str) -> f64 {
    relaxed_match(response, target)
}

/// Create the ChartQA task
pub fn chartqa() -> Task {
    Task::new("chartqa", samples, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relaxed_match_exact() {
        assert_eq!(relaxed_match("FINAL ANSWER: 42", "42"), 1.0);
        assert_eq!(relaxed_match("FINAL ANSWER: Technology", "technology"), 1.0);
    }

    #[test]
    fn test_relaxed_match_numeric_tolerance() {
        // 5% tolerance: 100 +/- 5 should match
        assert_eq!(relaxed_match("FINAL ANSWER: 102", "100"), 1.0);
        assert_eq!(relaxed_match("FINAL ANSWER: 98", "100"), 1.0);
        // Outside tolerance
        assert_eq!(relaxed_match("FINAL ANSWER: 110", "100"), 0.0);
    }

    #[test]
    fn test_relaxed_match_currency() {
        assert_eq!(relaxed_match("FINAL ANSWER: $1234", "$1,234"), 1.0);
        assert_eq!(relaxed_match("FINAL ANSWER: 35%", "35"), 1.0);
    }

    #[test]
    fn test_relaxed_match_zero() {
        assert_eq!(relaxed_match("FINAL ANSWER: 0", "0"), 1.0);
        assert_eq!(relaxed_match("FINAL ANSWER: 0.1", "0"), 0.0);
    }

    #[test]
    fn test_format_prompt() {
        let prompt = format_chartqa_prompt("What is the value?");
        assert!(prompt.contains("What is the value?"));
        assert!(prompt.contains("FINAL ANSWER"));
    }

    #[test]
    fn test_samples_load() {
        let samples = samples(Some(3), 42).unwrap();
        assert_eq!(samples.len(), 3);
        assert!(!samples[0].images.is_empty());
    }
}
