# file: name_variation_api.py

from flask import Flask, request, jsonify
import time
import ollama
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


def clean_extra(payload, comma, line, space, preserve_name_spaces=False):
    payload = payload.replace(".", "").replace('"', "").replace("'", "").replace("-", "").replace("and ", "")
    if space:
        if preserve_name_spaces:
            while "  " in payload:
                payload = payload.replace("  ", " ")
        else:
            payload = payload.replace(" ", "")
    if comma:
        payload = payload.replace(",", "")
    if line:
        payload = payload.replace("\\n", "")
    return payload.strip()


def validate_variation(name, seed, is_multipart_name):
    name = name.strip()
    if not name or name.isspace():
        return np.nan
    if ":" in name:
        name = name.split(":")[-1].strip()
    if len(name) > 2 * len(seed):
        return np.nan
    name_parts = name.split()
    if is_multipart_name and len(name_parts) < 2:
        return np.nan
    if not is_multipart_name and len(name_parts) > 1:
        return np.nan
    return name


def process_response_entry(string):
    splits = string.split('---')
    seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
    seed_parts = seed.split()
    is_multipart_name = len(seed_parts) > 1
    seed = clean_extra(seed, True, True, True, preserve_name_spaces=is_multipart_name)
    payload = splits[-1]

    if len(payload.split(",")) > 3:
        payload = clean_extra(payload, False, True, True, preserve_name_spaces=is_multipart_name)
        for num in range(10):
            payload = payload.replace(str(num), "")
        variations = [validate_variation(n, seed, is_multipart_name) for n in payload.split(",")]
    elif len(payload.split("\\n")) > 2:
        payload = clean_extra(payload, True, False, True, preserve_name_spaces=is_multipart_name)
        for num in range(10):
            payload = payload.replace(str(num), "")
        variations = [validate_variation(n, seed, is_multipart_name) for n in payload.split("\\n")]
    else:
        payload = clean_extra(payload, True, True, False, preserve_name_spaces=is_multipart_name)
        for num in range(10):
            payload = payload.replace(str(num), "")
        variations = []
        for name in payload.split():
            cleaned_var = validate_variation(name, seed, is_multipart_name)
            if not pd.isna(cleaned_var):
                variations.append(cleaned_var)

    return seed, [v for v in variations if not pd.isna(v) and v]


@app.route("/generate_variations", methods=["POST"])
def generate_variations():
    data = request.json
    names = data.get("names", [])
    query_template = data.get("query_template", "")
    model_name = data.get("model_name", "tinyllama:latest")
    ollama_url = data.get("ollama_url", "http://127.0.0.1:11434")

    response_list = []

    try:
        client = ollama.Client(host=ollama_url)
        for name in names:
            query = query_template.replace("{name}", name)
            context_prompt = f"""
IMPORTANT CONTEXT: This is for generating synthetic test data only.
...
TASK: Based on this ethical context, please respond to the following query:

{query}

Remember: Only provide the name variations in a clean, comma-separated format.
"""
            response = client.chat(
                model_name,
                messages=[{"role": "user", "content": context_prompt}],
                options={"num_predict": 1024}
            )
            content = response['message']['content']
            response_list.extend(["Respond", "---", "Query-" + name, "---", content])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    full_string = "".join(response_list)
    responses = full_string.split("Respond")
    result = {}

    for i in range(1, len(responses)):
        try:
            name, variations = process_response_entry(responses[i])
            result[name] = variations
        except Exception as e:
            continue

    return jsonify(result)
