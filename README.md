# Detecting Liberalism - Model Evaluation Framework 🔍

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Models-orange.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview 🎯

This project focuses on detecting and analyzing liberal viewpoints and expressions using various language models. The framework compares both open-source models (via Ollama) and closed-source models in their ability to identify and analyze liberal perspectives across different dimensions of text. The project employs systematic model evaluation, label generation, and performance analysis to understand how different AI models interpret and classify liberal content.

## Project Structure 📁

| Category        | File                  | Description                                                  |
| --------------- | --------------------- | ------------------------------------------------------------ |
| **Core Models** | `open_ai.py`          | Implementation for OpenAI model interactions and evaluations |
|                 | `ollama_run_async.py` | Asynchronous implementation for running Ollama models        |
|                 | `olllama-models.py`   | Core implementations for Ollama model handling               |
| **Evaluation**  | `eval-metrics.py`     | Scripts for calculating and analyzing evaluation metrics     |
| **Utilities**   | `add_labels.py`       | Utilities for label generation and management                |
|                 | `fix_json_load.py`    | JSON parsing and fixing utilities                            |

### Data Files 📊

| File                                                 | Purpose                                            |
| ---------------------------------------------------- | -------------------------------------------------- |
| `all_labels.csv`                                     | Comprehensive dataset of labels                    |
| `examples_with_labels.csv`                           | Example dataset with associated labels             |
| `sample_data_labelled.csv`                           | Sample labeled dataset for testing                 |
| `dimensions_definitions_examples_and_prompt_new.csv` | Definitions and examples for evaluation dimensions |

### Visualization 📈

| Visualization                   | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `agreement_metrics_heatmap.png` | Heatmap visualization of model agreement metrics |
| `f1_heatmap_all_models.png`     | F1 score comparisons across different models     |

## Key Features ⭐

- 🚀 Asynchronous model evaluation
- 🤖 Support for multiple model providers (Ollama, OpenAI)
- 📊 Comprehensive metrics calculation and visualization
- 🔄 JSON response parsing and validation
- 🏷️ Label generation and management
- 📈 Performance comparison across models

## Requirements 🛠️

| Requirement | Version/Details        |
| ----------- | ---------------------- |
| Python      | 3.x or higher          |
| Ollama      | Latest version         |
| OpenAI API  | Valid API key required |

## Usage 📝

1. 📥 Ensure all required dependencies are installed
2. 🔑 Configure API keys and model access
3. 🚀 Use the appropriate script based on your evaluation needs:

   ```bash
   # For Ollama model evaluation
   python ollama_run_async.py

   # For OpenAI model evaluation
   python open_ai.py

   # For analyzing results
   python eval-metrics.py
   ```

## Data Structure 📊

| Dataset Type       | Contents                                   |
| ------------------ | ------------------------------------------ |
| Training Examples  | Curated text samples for model training    |
| Labeled Data       | Annotated datasets with liberal indicators |
| Evaluation Metrics | Performance measurements and comparisons   |
| Model Results      | Comparative analysis outputs               |

## Visualization Capabilities 📊

- 📈 Agreement metrics between models
- 📊 F1 score comparisons
- 🔥 Performance heatmaps

## Security Note 🔒

- 🔐 API keys and sensitive credentials should be stored securely
- ⚠️ Refer to `key.txt` for API configuration (ensure this is not committed to version control)

## Contributing 🤝

Please follow the project's coding standards and documentation practices when contributing new features or fixes. We welcome contributions that enhance the project's ability to detect and analyze liberal perspectives.

## License 📄

[MIT License](https://opensource.org/licenses/MIT)

---

_Made with ❤️ for advancing political discourse analysis through AI_
