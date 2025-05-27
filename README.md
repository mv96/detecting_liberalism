# Project Detecting Liberalism

A comprehensive text classification system that leverages various language models (including OpenAI and Ollama API) for multi-label classification tasks.

## Contributors

### Elena Cossu

- **Institution**: SciencesPo Paris
- **Position**: Postdoctoral Researcher at Centre for European Studies and Comparative Politics (CEE)
- **Research Focus**: Data science and quasi-experimental methods in political science, measuring democratic backsliding in Europe
- **Email**: elena.cossu@sciencespo.fr
- **Profile**: [SciencesPo Profile](https://www.sciencespo.fr/centre-etudes-europeennes/en/directory/cossu-elena/)

### Shrey Mishra

- **Institution**: École Normale Supérieure (ENS)
- **Position**: PhD Student in Machine Learning and Optimization
- **Research Focus**: Information extraction from scientific articles, AI/ML techniques
- **Email**: mishra@di.ens.fr
- **Profile**: [PRAIRIE Institute Profile](https://prairie-institute.fr/chairs/mishra-shrey/)

### Jean-Philippe Cointet

- **Institution**: SciencesPo Paris
- **Position**: Professor of Sociology and Director of the Open Institute for Digital Transformations
- **Research Focus**: Computational social sciences, text analysis, network science
- **Email**: jeanphilippe.cointet@sciencespo.fr
- **Profile**: [médialab Profile](https://medialab.sciencespo.fr/en/people/jean-philippe-cointet/)

## Project Overview

This project implements a text classification system that can:

- Process text inputs through multiple language models
- Apply multi-label classification across various dimensions
- Evaluate and compare model performance
- Handle batch processing of large datasets
- Support both closed and open-source language models

## Key Features

- Multi-model support (OpenAI (using batch API), Ollama, Mistral-7B)
- Batch processing capabilities
- Asynchronous processing support
- Comprehensive evaluation metrics
- JSON response parsing and validation
- Performance visualization tools

## Project Structure

```
project/
├── open_ai.py              # OpenAI integration and batch processing
├── ollama_run_async.py     # Asynchronous processing with Ollama
├── olllama-models.py       # Ollama model implementations
├── eval-metrics.py         # Evaluation metrics calculation
├── add_labels.py           # Label addition utilities
├── fix_json_load.py        # JSON parsing and fixing utilities
└── test.py                 # Testing utilities
```

## Setup

1. Clone the repository
2. Install dependencies (requirements will be added soon)
3. Set up your API keys:
   - Create a `key.txt` file with your OpenAI API key
   - Configure any necessary Ollama settings

## Usage

The project supports multiple modes of operation:

1. **OpenAI Integration**:

   ```python
   python open_ai.py
   ```

2. **Ollama Processing**:

   ```python
   python ollama_run_async.py
   ```

3. **Evaluation**:
   ```python
   python eval-metrics.py
   ```

## Data

The project works with several data formats:

- CSV files for input data
- JSONL files for batch processing
- JSON files for model outputs
- Visualization outputs (PNG files for heatmaps)

## Visualization

The project includes visualization tools for:

- Agreement metrics heatmaps
- F1 score comparisons across models
- Other performance metrics

## Contributing

Contributions are welcome please contact the authors and make a pull request upon approval.

## License

Copyright © 2024 Shrey Mishra, Elena Cossu, Jean-Philippe Cointet. All rights reserved.

This work is proprietary and confidential. No part of this project, including but not limited to the code, documentation, data, models, and methodologies, may be reproduced, distributed, or transmitted in any form or by any means, without the prior written permission of ALL copyright holders. Any usage, modification, or distribution of this work requires explicit written consent from ALL three authors: Elena Cossu, Shrey Mishra, and Jean-Philippe Cointet.

Unauthorized copying, modification, merger, publication, distribution, sublicensing, and/or selling of this software and associated documentation is strictly prohibited. No single author or subset of authors may grant permissions independently - unanimous agreement from all three authors is required for any and all uses beyond the scope of this research collaboration.
