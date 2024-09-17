#
# Copyright (c) 2024 Educa, https://www.educa.ch
#
# This file is part of Educa24 
# (see https://www.educa.ch/de/veranstaltungen/2024/educa24-daten-als-grundlage-fuer-ki-systeme).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import argparse
from pathlib import Path
import time
import json


def get_current_time():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def transcribe(file, model_file):
    import torch  # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    import whisper # pip install -U openai-whisper

    assert Path(file).is_file()
    model = None
    cuda = False
    if torch.cuda.is_available():
        print("Transcription on GPU ...")
        model = whisper.load_model(model_file, device="cuda")
        cuda = True
    else:
        print("Transcription on CPU ...")
        model = whisper.load_model(model_file)

    time_start = time.time()

    result = model.transcribe(file)

    time_end = time.time()
    time_duration = time_end - time_start

    data = {'model': model_file, "cuda": cuda, "duration": time_duration, "input": file, "result": result}

    json_file = file + ".json"
    with open(json_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return data


def get_prompts(language):
    from langchain_core.prompts import PromptTemplate

    map_prompt = None
    reduce_prompt = None

    if language == "fr":
        map_prompt_template = """Résume cette présentation en 5 phrases concises:

        {text}

        RÉSUMÉ CONCIS:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        reduce_prompt_template = """Résume ce texte en 20 phrases concises:

        {text}

        RÉSUMÉ CONCIS:"""

        reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])
    
    elif language == "de":
        map_prompt_template = """Fasse diesen Vortrag in 5 prägnanten Sätzen zusammen:

        {text}

        PRÄGNANTE ZUSAMMENFASSUNG:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        reduce_prompt_template = """Fasse diesen Text in 20 prägnanten Sätzen zusammen:

        {text}

        PRÄGNANTE ZUSAMMENFASSUNG:"""

        reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])

    else:
        if language != "en":
            print(f"Unsupported language detected ({language}), defaulting to english.")

        map_prompt_template = """Write a concise summary of the following presentation in 5 sentences:

        {text}

        CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        reduce_prompt_template = """Summarize the following into one concise summary of 20 sentences:

        {text}

        CONCISE SUMMARY:"""

        reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])


    return[map_prompt, reduce_prompt]


def summarize(file, model, chain_type="stuff", temperature=0, verbose=False):
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.llms import Ollama
    from langchain_core.documents import Document
    import tiktoken

    print(f"Summarize using {model}, chain type = {chain_type}, temperature={temperature}")

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    time_start = time.time()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1024, chunk_overlap=128)

    document = Document(page_content=data['result']['text'], metadata={"source": file})
    split_docs = text_splitter.split_documents([document])
    print(f"Number of chunks: {len(split_docs)}")

    llm = Ollama(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
        keep_alive = 0)

    [map_prompt, reduce_prompt] = get_prompts(data['result']['language'])

    chain = load_summarize_chain(llm, chain_type=chain_type, map_prompt=map_prompt, combine_prompt=reduce_prompt, verbose=verbose)

    output = chain.invoke(split_docs)

    time_end = time.time()
    time_duration = time_end - time_start

    if 'summary' not in data.keys():
        data['summary'] = {}
        
    dt_string = "_" + get_current_time()
    data['summary'][model+dt_string] = {"duration": time_duration,
                                        "chain_type": chain_type,
                                        "temperature": temperature,
                                        "map_prompt": map_prompt.template,
                                        "reduce_prompt": reduce_prompt.template,
                                        "response": output['output_text']}

    with open(file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)


def slice_audio(input, output, start_time, stop_time):
    from pydub import AudioSegment

    print(f"Slicing audio file {input} into {output}.")
    start_second = None
    duration = None

    if start_time != "":
        start_second = time_to_seconds(start_time)

    if stop_time != "":
        if start_second == "":
            duration = time_to_seconds(stop_time)
        else:
            duration = time_to_seconds(stop_time) - start_second

    audio = AudioSegment.from_file(input, start_second=start_second, duration=duration, format="mp3")

    with open(output, "wb") as f:
        audio.export(output, format="mp3")


def model_parse(m, args):
    if m=="llama3.1:8b":
        return args.llama3_1
    elif m=="mistral:7b":
        return args.mistral
    elif m=="mistral-nemo:12b":
        return args.mistral_nemo
    else:
        assert False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input audio file", type=str)
    parser.add_argument("--whisper_model", type=str, choices=["large", "medium", "small"], default="large", help="define the model file for whisper transcription")
    parser.add_argument("--llama3_1", action="store_true")
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--mistral_nemo", action="store_true")
    parser.add_argument("--temperature", type=float, default=0, help="sets the temperature, higher is more creative, lower is more coherent")
    parser.add_argument("--transcribe", action='store_true')
    parser.add_argument("--repeat", type=int, default=1, help="repeat the summarization step n times")
    parser.add_argument("--start_time", type=str, default="", help="start time for slicing the audio file in format hh:mm:ss")
    parser.add_argument("--stop_time", type=str, default="", help="stop time for slicing the audio file in format hh:mm:ss")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    assert Path(args.input).is_file()
    assert args.temperature >= 0 and args.temperature <= 1
    assert args.repeat >= 1

    audio_file = args.input
    if args.start_time != "" or args.stop_time != "":
        timestamp = get_current_time()
        if Path(audio_file).suffix == ".mp3":
            audio_file = audio_file.replace(".mp3", "." + timestamp + ".mp3")
        else:
            audio_file = args.input + "." + timestamp + ".mp3"

        slice_audio(args.input, audio_file, args.start_time, args.stop_time)

    if args.transcribe:
        transcribe(audio_file, args.whisper_model)

    models = ['mistral-nemo:12b', 'mistral:7b', 'llama3.1:8b']

    model_count = 0

    for m in models:
        if not model_parse(m, args):
            if m != models[-1]:
                continue
            else:
                if model_count > 0:
                    continue

        model_count = model_count + 1
        for i in range(0, args.repeat):
            print(m)
            summarize(audio_file + '.json', m, chain_type="map_reduce", temperature=args.temperature, verbose=args.verbose)


if __name__ == "__main__":
    main()
