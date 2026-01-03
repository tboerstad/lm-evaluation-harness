//! Task registry and implementations

pub mod chartqa;
pub mod gsm8k;

use crate::core::Task;
use crate::error::{Result, TinyEvalError};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Task factory function type
type TaskFactory = fn() -> Task;

/// Registry of available tasks
static TASK_REGISTRY: Lazy<HashMap<&'static str, TaskFactory>> = Lazy::new(|| {
    let mut m: HashMap<&'static str, TaskFactory> = HashMap::new();
    m.insert("gsm8k_llama", gsm8k::gsm8k_llama);
    m.insert("chartqa", chartqa::chartqa);
    m
});

/// Get a task by name
pub fn get_task(name: &str) -> Result<Task> {
    TASK_REGISTRY
        .get(name)
        .map(|factory| factory())
        .ok_or_else(|| {
            let available: Vec<&str> = TASK_REGISTRY.keys().copied().collect();
            TinyEvalError::UnknownTask(name.to_string(), available.join(", "))
        })
}

/// Get all available task names
pub fn available_tasks() -> Vec<&'static str> {
    TASK_REGISTRY.keys().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_task_gsm8k() {
        let task = get_task("gsm8k_llama").unwrap();
        assert_eq!(task.name, "gsm8k_llama");
    }

    #[test]
    fn test_get_task_chartqa() {
        let task = get_task("chartqa").unwrap();
        assert_eq!(task.name, "chartqa");
    }

    #[test]
    fn test_unknown_task() {
        let result = get_task("unknown");
        assert!(result.is_err());
        if let Err(TinyEvalError::UnknownTask(name, _)) = result {
            assert_eq!(name, "unknown");
        } else {
            panic!("Expected UnknownTask error");
        }
    }

    #[test]
    fn test_available_tasks() {
        let tasks = available_tasks();
        assert!(tasks.contains(&"gsm8k_llama"));
        assert!(tasks.contains(&"chartqa"));
    }
}
