# AI_Templates

AI_Templates is a Python package designed to simplify and streamline your work with the OpenAI API. It provides Python typing support, error checking, and a usage meter to help manage API costs. Additionally, AI_Templates offers built-in examples of using ChromaDB and tools for efficient prompt engineering with OpenAI.

## Features

- **Python Typing Support**: Enjoy the benefits of Python's dynamic typing system while using OpenAI API.
- **Error Checking**: Automatically catch and handle errors during API calls.
- **Usage Meter**: Keep track of your OpenAI API usage with a built-in metering system.
- **ChromaDB Integration**: Work directly with ChromaDB from the AI_Templates interface.
- **Asynchronous Chat Completions**: Use the `-asnc` flag to run asynchronous chat completions. The built-in `print_every` option prints every time a completion finishes in parallel. If a list is passed into the `response_list` attribute, it updates that list as completions finish. If you'd like to maintain the order of your completions, pass in the `keep_order` boolean as `True`.
- **Prompt Engineering Examples**: Get started quickly with included examples of prompt engineering techniques.

## Installation

Clone the repository and install the package by running `pip install -e .` in the repository directory.

## Creating Jupyter Notebook Templates

Create Jupyter notebook templates for prompt engineering with the following command:

```bash
ai_templates name_of_notebook
```

Include a Chroma database in the template with the `-db` flag:

```bash
ai_templates name_of_notebook -db
```

For asynchronous chat completions, add the `-asnc` flag:

```bash
ai_templates name_of_notebook -db -asnc
```

## Documentation

The `/notebooks` directory contains various example notebooks that demonstrate the usage of the different classes and methods provided by AI_Templates:

- `oai_examples.ipynb`: Provides examples for the OpenAI API.
- `chroma_examples.ipynb`: Demonstrates usage of ChromaDB.
- `prompt_engineering_example.ipynb`: A comprehensive guide on prompt engineering, including various techniques, usage of ChromaDB and the OpenAI library together.

## Requirements

- Be sure to have a `.env` file with the `OPENAI_API_KEY` defined in the root of your project. AWS env variables are for ChromaDB deployment https://docs.trychroma.com/deployment
- This library doesn't currently support the use of top-p and text models like `text-davinci-003`. Streaming and logit_bias parameters are not supported

## Contributing

I'd welcome and appreciate contributions to AI_Templates. If you'd like to contribute a feature, please submit a pull request. For bugs, please open a new issue, and I'll address it promptly.
