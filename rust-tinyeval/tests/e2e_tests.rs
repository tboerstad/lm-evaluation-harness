//! End-to-end tests for tinyeval CLI

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;
use wiremock::matchers::{body_json_schema, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Helper to create a mock OpenAI API response
fn mock_chat_completion_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })
}

#[tokio::test]
async fn test_gsm8k_evaluation_outputs_json() {
    let mock_server = MockServer::start().await;

    // Mock the chat completions endpoint
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(1..)
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama",
        "--model-args",
        &format!("model=test-model,base_url={}/v1", mock_server.uri()),
        "--max-samples",
        "1",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should output valid JSON
    let result: serde_json::Value = serde_json::from_str(&stdout)
        .expect("Output should be valid JSON");

    // Check expected structure
    assert!(result.get("results").is_some());
    assert!(result.get("dataset_hash").is_some());
    assert!(result.get("total_seconds").is_some());
    assert!(result.get("config").is_some());

    // Check task results
    let gsm8k_result = &result["results"]["gsm8k_llama"];
    assert!(gsm8k_result.get("task").is_some());
    assert!(gsm8k_result.get("task_hash").is_some());
    assert!(gsm8k_result.get("metrics").is_some());
    assert!(gsm8k_result.get("num_samples").is_some());
    assert!(gsm8k_result.get("elapsed").is_some());
}

#[tokio::test]
async fn test_gen_kwargs_passed_to_api() {
    let mock_server = MockServer::start().await;

    // Set up mock that verifies request body contains generation kwargs
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({
            "temperature": 0.7,
            "max_tokens": 100
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 42")),
        )
        .expect(1..)
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama",
        "--model-args",
        &format!("model=test-model,base_url={}/v1", mock_server.uri()),
        "--gen-kwargs",
        "temperature=0.7,max_tokens=100",
        "--max-samples",
        "1",
    ]);

    cmd.assert().success();
}

#[tokio::test]
async fn test_invalid_task_raises_error() {
    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "nonexistent_task",
        "--model-args",
        "model=test,base_url=http://localhost:8000/v1",
    ]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Unknown task"));
}

#[tokio::test]
async fn test_output_path_writes_results_json() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(1..)
        .mount(&mock_server)
        .await;

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path();

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama",
        "--model-args",
        &format!("model=test-model,base_url={}/v1", mock_server.uri()),
        "--max-samples",
        "1",
        "--output-path",
        output_path.to_str().unwrap(),
    ]);

    cmd.assert().success();

    // Verify results.json was written
    let results_file = output_path.join("results.json");
    assert!(results_file.exists(), "results.json should be created");

    let contents = fs::read_to_string(&results_file).unwrap();
    let result: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert!(result.get("results").is_some());
}

#[tokio::test]
async fn test_log_samples_writes_jsonl() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(1..)
        .mount(&mock_server)
        .await;

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path();

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama",
        "--model-args",
        &format!("model=test-model,base_url={}/v1", mock_server.uri()),
        "--max-samples",
        "2",
        "--output-path",
        output_path.to_str().unwrap(),
        "--log-samples",
    ]);

    cmd.assert().success();

    // Verify JSONL file was written
    let jsonl_file = output_path.join("samples_gsm8k_llama.jsonl");
    assert!(jsonl_file.exists(), "samples JSONL should be created");

    let contents = fs::read_to_string(&jsonl_file).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 2, "Should have 2 sample lines");

    // Each line should be valid JSON
    for line in lines {
        let sample: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(sample.get("doc_id").is_some());
        assert!(sample.get("target").is_some());
        assert!(sample.get("response").is_some());
        assert!(sample.get("exact_match").is_some());
    }
}

#[tokio::test]
async fn test_model_args_passed_to_config() {
    let mock_server = MockServer::start().await;

    // Verify the model name is passed correctly
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({
            "model": "my-custom-model"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(1..)
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama",
        "--model-args",
        &format!("model=my-custom-model,base_url={}/v1", mock_server.uri()),
        "--max-samples",
        "1",
    ]);

    cmd.assert().success();
}

#[tokio::test]
async fn test_multiple_tasks() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(2..)  // At least 2 requests for 2 tasks
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--tasks",
        "gsm8k_llama,gsm8k_llama",  // Run same task twice to test multiple
        "--model-args",
        &format!("model=test-model,base_url={}/v1", mock_server.uri()),
        "--max-samples",
        "1",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stdout = String::from_utf8_lossy(&output.stdout);

    let result: serde_json::Value = serde_json::from_str(&stdout)
        .expect("Output should be valid JSON");

    // Both tasks should have results
    assert!(result["results"]["gsm8k_llama"].is_object());
}

#[tokio::test]
async fn test_seed_affects_reproducibility() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response("The final answer is 4")),
        )
        .expect(2..)
        .mount(&mock_server)
        .await;

    // Run with same seed twice and compare hashes
    let run_with_seed = |seed: u64| {
        let mut cmd = Command::cargo_bin("tinyeval").unwrap();
        cmd.args([
            "--tasks",
            "gsm8k_llama",
            "--model-args",
            &format!("model=test-model,base_url={}/v1", mock_server.uri()),
            "--max-samples",
            "1",
            "--seed",
            &seed.to_string(),
        ]);
        cmd
    };

    let output1 = run_with_seed(42).output().unwrap();
    let output2 = run_with_seed(42).output().unwrap();

    let result1: serde_json::Value =
        serde_json::from_slice(&output1.stdout).unwrap();
    let result2: serde_json::Value =
        serde_json::from_slice(&output2.stdout).unwrap();

    // Same seed should produce same task hash
    assert_eq!(
        result1["results"]["gsm8k_llama"]["task_hash"],
        result2["results"]["gsm8k_llama"]["task_hash"]
    );
}

#[test]
fn test_missing_required_args() {
    // Missing --tasks
    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args([
        "--model-args",
        "model=test,base_url=http://localhost:8000/v1",
    ]);
    cmd.assert().failure();

    // Missing --model-args
    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.args(["--tasks", "gsm8k_llama"]);
    cmd.assert().failure();
}

#[test]
fn test_help_flag() {
    let mut cmd = Command::cargo_bin("tinyeval").unwrap();
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--tasks"))
        .stdout(predicate::str::contains("--model-args"));
}
