//! tinyeval - A minimal harness for evaluating LLMs through OpenAI-compatible APIs

mod core;
mod error;
mod tasks;

use crate::core::{compute_task_hash, run_task, APIConfig, GenKwargs, LoggedSample, TaskResult};
use crate::error::{Result, TinyEvalError};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

/// A minimal, lean harness for evaluating LLMs through OpenAI-compatible APIs
#[derive(Parser, Debug)]
#[command(name = "tinyeval")]
#[command(version = "1.0.0")]
#[command(about = "Evaluate LLMs through OpenAI-compatible APIs")]
struct Args {
    /// Comma-separated list of tasks to run
    #[arg(long, required = true)]
    tasks: String,

    /// Model configuration: model=name,base_url=url[,seed=N,num_concurrent=N,timeout=N,max_retries=N,api_key=key]
    #[arg(long, required = true)]
    model_args: String,

    /// Generation kwargs: temperature=N,max_tokens=N,top_p=N,...
    #[arg(long, default_value = "")]
    gen_kwargs: String,

    /// Maximum samples per task
    #[arg(long)]
    max_samples: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output directory for results
    #[arg(long)]
    output_path: Option<PathBuf>,

    /// Log individual samples to JSONL files
    #[arg(long, default_value = "false")]
    log_samples: bool,
}

/// Overall evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvalResults {
    results: HashMap<String, TaskResultOutput>,
    dataset_hash: String,
    total_seconds: f64,
    config: ConfigOutput,
}

/// Task result for output (without samples in main results)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskResultOutput {
    task: String,
    task_hash: String,
    metrics: crate::core::Metrics,
    num_samples: usize,
    elapsed: f64,
}

impl From<&TaskResult> for TaskResultOutput {
    fn from(result: &TaskResult) -> Self {
        Self {
            task: result.task.clone(),
            task_hash: result.task_hash.clone(),
            metrics: result.metrics.clone(),
            num_samples: result.num_samples,
            elapsed: result.elapsed,
        }
    }
}

/// Configuration output
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigOutput {
    model: String,
    max_samples: Option<usize>,
    seed: u64,
    tasks: Vec<String>,
}

/// Run evaluation for all specified tasks
async fn evaluate(
    task_names: &[String],
    config: &APIConfig,
    gen_kwargs: &GenKwargs,
    max_samples: Option<usize>,
    log_samples: bool,
    output_path: Option<&PathBuf>,
) -> Result<EvalResults> {
    let start = Instant::now();
    let mut results: HashMap<String, TaskResult> = HashMap::new();
    let mut all_samples: Vec<crate::core::Sample> = Vec::new();

    for task_name in task_names {
        let task = tasks::get_task(task_name)?;

        // Load samples for hashing
        let samples = task.load_samples(max_samples, config.seed)?;
        all_samples.extend(samples);

        // Run task
        let result = run_task(&task, config, gen_kwargs, max_samples, log_samples).await?;

        // Write samples JSONL if requested
        if log_samples {
            if let Some(ref path) = output_path {
                write_samples_jsonl(path, &task.name, &result.samples)?;
            }
        }

        results.insert(task_name.clone(), result);
    }

    // Compute overall dataset hash
    let dataset_hash = compute_task_hash(&all_samples)?;

    let total_seconds = start.elapsed().as_secs_f64();

    let eval_results = EvalResults {
        results: results.iter().map(|(k, v)| (k.clone(), v.into())).collect(),
        dataset_hash,
        total_seconds,
        config: ConfigOutput {
            model: config.model.clone(),
            max_samples,
            seed: config.seed,
            tasks: task_names.to_vec(),
        },
    };

    // Write results.json if output path specified
    if let Some(ref path) = output_path {
        write_results_json(path, &eval_results)?;
    }

    Ok(eval_results)
}

/// Write results to JSON file
fn write_results_json(output_path: &PathBuf, results: &EvalResults) -> Result<()> {
    fs::create_dir_all(output_path)?;
    let file_path = output_path.join("results.json");
    let file = File::create(&file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, results)?;
    Ok(())
}

/// Write samples to JSONL file
fn write_samples_jsonl(
    output_path: &PathBuf,
    task_name: &str,
    samples: &[LoggedSample],
) -> Result<()> {
    fs::create_dir_all(output_path)?;
    let file_path = output_path.join(format!("samples_{}.jsonl", task_name));
    let file = File::create(&file_path)?;
    let mut writer = BufWriter::new(file);

    for sample in samples {
        serde_json::to_writer(&mut writer, sample)?;
        writeln!(writer)?;
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();

    // Parse task names
    let task_names: Vec<String> = args
        .tasks
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if task_names.is_empty() {
        return Err(TinyEvalError::InvalidModelArgs(
            "No tasks specified".to_string(),
        ));
    }

    // Validate tasks exist before running
    for task_name in &task_names {
        tasks::get_task(task_name)?;
    }

    // Parse model config
    let mut config = APIConfig::from_model_args(&args.model_args)?;
    config.seed = args.seed;

    // Parse generation kwargs
    let gen_kwargs = if args.gen_kwargs.is_empty() {
        GenKwargs::default()
    } else {
        GenKwargs::from_str(&args.gen_kwargs)?
    };

    // Run evaluation
    let results = evaluate(
        &task_names,
        &config,
        &gen_kwargs,
        args.max_samples,
        args.log_samples,
        args.output_path.as_ref(),
    )
    .await?;

    // Output results to stdout
    let json = serde_json::to_string_pretty(&results)?;
    println!("{}", json);

    Ok(())
}
