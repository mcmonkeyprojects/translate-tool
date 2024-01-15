extern crate intel_mkl_src;

use std::io::Write;
use std::path::PathBuf;

use candle_transformers::models::quantized_t5::{self as t5, T5ForConditionalGeneration};

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, api::sync::ApiRepo, Repo, RepoType};
use tokenizers::{Tokenizer, TokenizerImpl, ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper};

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// optional verbose debug log
    #[arg(long, default_value = "false")]
    verbose: bool,

    /// JSON source file
    #[arg(long)]
    in_json: String,

    /// JSON result file
    #[arg(long)]
    out_json: String,

    // Language code eg 'en', 'de', etc.
    #[arg(long)]
    language: String,
    
    /// The relative ratio of output vs input after which the model is presumed to have broken.
    #[arg(long, default_value_t = 5.0)]
    max_ratio: f32,

    // All args below are from HF example, (with some added defaults and docs).

    /// The model repository to use on the HuggingFace hub.
    #[arg(long, default_value = "jbochi/madlad400-7b-mt-bt")]
    model_id: String,

    /// HF example didn't document this but I think it's the branch
    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "model-q4k.gguf")]
    weight_file: Option<String>,

    /// HF didn't document this either but it lets you switch to a different config if the repo is missing one
    #[arg(long)]
    config_file: Option<String>,

    /// Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let device = Device::Cpu;
        let repo = Repo::with_revision(args.model_id.to_owned(), RepoType::Model, args.revision.to_owned());
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &args.config_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => api.get("config.json")?
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &args.weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => api.get("model.gguf")?
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename)?;
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}

fn verbose(args: &Args, msg: &str) {
    if args.verbose {
        println!("{}", msg);
    }
}

fn sane_split(prompt: &str, split: &str, min: usize, max: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    for word in prompt.split(split) {
        if !current.is_empty() && current.len() + word.len() + split.len() > max {
            result.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push_str(split);
        }
        current.push_str(word);
        if current.len() >= min {
            result.push(current);
            current = String::new();
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn translate(args: Args, builder: &T5ModelBuilder, model: &mut T5ForConditionalGeneration, tokenizer: &TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>, logits_processor: &mut LogitsProcessor, in_prompt: &str, language_token: &Vec<u32>) -> Result<(String, usize)> {
    let prompt = in_prompt.trim();
    verbose(&args, format!("mcmonkey's Translate-Tool received input '{}', tokenizing...", prompt).as_str());
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();

    // Autosplitter to accomodate the 128 token limit.
    let mut subtranslate_chunk = |parts: Vec<String>, combine: &str| -> Result<(String, usize)> {
        let mut result_str = "".to_owned();
        let mut total_len: usize = 0;
        for part in parts {
            let (part_result, part_len) = translate(args.clone(), builder, model, tokenizer, logits_processor, &part, language_token)?;
            if part_len == 0 {
                return Ok((part_result, 0));
            }
            total_len += part_len;
            if !result_str.is_empty() {
                result_str.push_str(combine);
            }
            result_str.push_str(&part_result);
        }
        result_str = result_str.trim().to_owned();
        Ok((result_str, total_len))
    };
    // Tries newlines, then sentence boundaries, then word boundaries, then gives up and uses raw token chunking.
    if tokens.len() > 80 {
        let chunks = sane_split(prompt, "\n", 0, 60);
        if chunks.len() > 1 {
            verbose(&args, format!("Prompt length is {}, splitting to {} chunks based on new line", tokens.len(), chunks.len()).as_str());
            return subtranslate_chunk(chunks, "\n");
        }
    }
    if tokens.len() > 80 {
        let chunks = sane_split(prompt, ". ", 0, 60);
        if chunks.len() > 1 {
            verbose(&args, format!("Prompt length is {}, splitting to {} chunks based on sentence boundaries", tokens.len(), chunks.len()).as_str());
            return subtranslate_chunk(chunks, ". ");
        }
    }
    if tokens.len() > 80 {
        let chunks = sane_split(prompt, ",", 30, 60);
        if chunks.len() > 1 {
            verbose(&args, format!("Prompt length is {}, splitting to {} chunks based on commas", tokens.len(), chunks.len()).as_str());
            return subtranslate_chunk(chunks, ",");
        }
    }
    if tokens.len() > 80 {
        let chunks = sane_split(prompt, " ", 30, 60);
        if chunks.len() > 1 {
            verbose(&args, format!("Prompt length is {}, splitting to {} chunks based on spaces", tokens.len(), chunks.len()).as_str());
            return subtranslate_chunk(chunks, " ");
        }
    }
    if tokens.len() > 80 {
        let mut result = "".to_owned();
        let mut total_len: usize = 0;
        verbose(&args, format!("Prompt length is {}, splitting to chunks around raw token split 50 (THIS IS BAD, WILL MISTRANSLATE NEAR BOUNDARY)", tokens.len()).as_str());
        for chunk in tokens.chunks(50) {
            let chunk = chunk.to_vec();
            let prompt = tokenizer.decode(chunk.as_slice(), false).map_err(E::msg)?;
            let (chunk_result, chunk_len) = translate(args.clone(), builder, model, tokenizer, logits_processor, &prompt, language_token)?;
            total_len += chunk_len;
            result.push_str(&chunk_result);
        }
        return Ok((result, total_len));
    }

    let mut actual_tokens = language_token.clone();
    actual_tokens.extend(tokens);

    model.clear_kv_cache();
    let input_token_ids = Tensor::new(&actual_tokens[..], &builder.device)?.unsqueeze(0)?;
    let mut output_token_ids = [builder.config.decoder_start_token_id.unwrap_or(builder.config.pad_token_id) as u32].to_vec();
    
    verbose(&args, "mcmonkey's Translate-Tool encoding input...");
    let encoder_output = model.encode(&input_token_ids)?;
    
    verbose(&args, "mcmonkey's Translate-Tool will now translate...");
    let start = std::time::Instant::now();
    let mut final_result_str: String = "".to_owned();
    for index in 0.. {
        if output_token_ids.len() > 512 {
            break;
        }
        let decoder_token_ids = if index == 0 || !builder.config.use_cache {
            Tensor::new(output_token_ids.as_slice(), &builder.device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], &builder.device)?.unsqueeze(0)?
        };
        let logits = model.decode(&decoder_token_ids, &encoder_output)?.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &output_token_ids[start_at..],
            )?
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == builder.config.eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            if args.verbose {
                print!("{text}");
            }
            std::io::stdout().flush()?;
            final_result_str.push_str(&text);
            if (final_result_str.len() as f32) / (prompt.len() as f32) > args.max_ratio {
                println!("WARNING: Result is {} chars long, but input was {} chars long - probably translation failure -- {prompt}, {final_result_str}", final_result_str.len(), prompt.len());
                return Ok(("".to_owned(), 0));
            }
        }
    }
    final_result_str = final_result_str.trim().to_owned();
    let dt = start.elapsed();
    verbose(&args, &format!("\n{} tokens generated ({:.2} token/s)\n", output_token_ids.len(), output_token_ids.len() as f64 / dt.as_secs_f64()));
    Ok((final_result_str, output_token_ids.len()))
}

fn main() -> Result<()> {
    println!("mcmonkey's Translate-Tool prepping...");
    let args = Args::parse();
    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let tokenizer = tokenizer.with_padding(None).with_truncation(None).map_err(E::msg)?;
    let temperature = if args.temperature <= 0. { None } else { Some(args.temperature) };
    let mut logits_processor = LogitsProcessor::new(299792458, temperature, args.top_p); // wtf is this handwritten static seed? (straight from HF example)
    let language_token = tokenizer.encode(format!("<2{}>", args.language).as_str(), false).map_err(E::msg)?.get_ids().to_vec();
    println!("mcmonkey's Translate-Tool loading model...");
    let mut model = builder.build_model()?;
    println!("mcmonkey's Translate-Tool JSON source file...");
    let json = std::fs::read_to_string(&args.in_json)?;
    let json: serde_json::Value = serde_json::from_str(&json)?;
    let json = json.as_object().ok_or(E::msg("JSON file must be an object"))?;
    let json = json.get("keys").ok_or(E::msg("JSON file must have a 'keys' field"))?;
    let json = json.as_object().ok_or(E::msg("JSON file 'keys' field must be an object"))?;
    println!("mcmonkey's Translate-Tool translating...");
    let mut total_len: usize = 0;
    let start = std::time::Instant::now();
    let mut count: usize = 0;
    let mut json_mut = json.clone();
    for (key, value) in json {
        let before_time = start.elapsed().as_secs_f64();
        let key = key.as_str();
        let value = value.as_str().ok_or(E::msg("JSON file 'keys' field must be an object of strings"))?;
        if !value.is_empty() {
            continue;
        }
        let (result_str, outlen) = translate(args.clone(), &builder, &mut model, tokenizer, &mut logits_processor, &key, &language_token)?;
        total_len += outlen;
        let after_time = start.elapsed().as_secs_f64();
        let time_current = outlen as f64 / (after_time - before_time);
        let time_total = total_len as f64 / after_time;
        count += 1;
        println!("translated [{}]: '{}' to '{}' - {} tokens generated ({:.2} token/s), total {} tokens ({:.2} token/s)", count, key, result_str, outlen, time_current, total_len, time_total);
        let value = serde_json::Value::String(result_str);
        json_mut.insert(key.to_owned(), value);
        if count % 20 == 19 {
            println!("mcmonkey's Translate-Tool saving intermediate result...");
            let json = serde_json::Value::Object(json_mut.clone());
            let mut json = serde_json::to_string_pretty(&json)?;
            json.push_str("\n");
            std::fs::write(&args.out_json, json)?;
        }
    }
    println!("Translated a total of {} entries out of {} available (skipping {}) after {:.2} seconds", count, json.len(), json.len() - count, start.elapsed().as_secs_f64());
    println!("mcmonkey's Translate-Tool saving result...");
    let json = serde_json::Value::Object(json_mut);
    let mut json = serde_json::to_string_pretty(&json)?;
    json.push_str("\n");
    std::fs::write(&args.out_json, json)?;
    println!("mcmonkey's Translate-Tool done!");
    Ok(())
}
