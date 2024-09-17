# Educa24
This repository contains a python script that is used as a demonstrator during the [Educa24 symposium](https://www.educa.ch/de/veranstaltungen/2024/educa24-daten-als-grundlage-fuer-ki-systeme) on September 18, 2024. It generates fully automatically a text summary from an audio recording. The script uses exclusively open-source tools and open-weights LLMs, which can be run on a local computing infrastructure. If the application runs locally, users retain complete control over their data at all times.

## Dependencies
To get started, setup a [Python](https://www.python.org/) environment on your infrastructure. We recommend using a computer with a rather high-end GPU.

The main dependencies are
- [Whisper](https://github.com/openai/whisper) for the speech-to-text part
- [LangChain](https://www.langchain.com/langchain) and [Ollama](https://ollama.com/) for the summary
  - Make sure the target LLMs, e.g. [Llama 3.1](https://www.llama.com/) and/or [Mistral-Nemo](https://mistral.ai/news/mistral-nemo/), are downloaded before you start, see [Ollama Models](https://ollama.com/library).

## Running
A typical processing of an audiofile `<my_audio.mp3>` is started with

`python summary_from_audio.py my_audio.mp3 --transcribe --llama3_1`

The result is written into a file `<my_audio.mp3.json>` which contains the transcription results at the beginning and the summarized text at the end.

The summarization step can be repeated on the same file without rerunning the transcription, e.g.:

`python summary_from_audio.py my_audio.mp3 --mistral_nemo`

The new result is appended to `<my_audio.mp3.json>`.

