"""
1. Zero-shot and few-shot learning
2. Log template extraction
3. Embedding generation for log templates

Author: Trung Doan Le, Soumya Sistla
"""

import os
import re
import json
from typing import List, Dict
from pydantic import BaseModel
from openai import OpenAI


#   export OPENAI_API_KEY="your_api_key_here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------
# Utility Classes and Functions
# --------------------------

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]


def extract_template_and_variables(log_line: str) -> Dict:
    """Zero-shot log template and variable extractor."""
    prompt = f"""
    You are a log parser. For the given log line:
    1. Identify a generalized log template by replacing variable parts (timestamp, usernames, IPs, IDs, numbers, etc.) with clearly typed placeholders like <TIMESTAMP>, <USERNAME>, <IP_ADDRESS>.
    2. Extract the values of those variables in a JSON dictionary.
    3. Output the JSON object with two fields: "template" and "variables".

    Log Line: {log_line}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"template": "<PARSE_ERROR>", "variables": {}, "error": str(e)}


def extract_template_and_variables_fewshot(log_line: str, few_shot_examples: str) -> Dict:
    """Few-shot log template and variable extractor."""
    prompt = f"""
    You are a log parser. For the given log line:
    1. Identify a generalized log template by replacing variable parts (timestamp, usernames, IPs, IDs, numbers, etc.) with clearly typed placeholders like <TIMESTAMP>, <USERNAME>, <IP_ADDRESS>.
    2. Extract the values of those variables in a JSON dictionary.
    3. Output the JSON object with two fields: "template", "variables", "original_log".

    If possible, make use of the examples to improve accuracy:

    {few_shot_examples}

    Now process the following:

    Log Line: {log_line}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()
    print(f"Raw model output:\n{content}\n")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"template": "<PARSE_ERROR>", "variables": {}, "error": str(e)}


def normalize_template(template: str) -> str:
    """Replace all placeholders with a uniform <*> format."""
    return re.sub(r"<[^>]+>", "<*>", template.strip())


def get_embedding(text: str) -> List[float]:
    """Get an embedding vector for a given text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


# --------------------------
# Example few-shot examples
# --------------------------

FEW_SHOT_EXAMPLES = """
Example 1:
Log Line:
2023-01-01 10:23:45 User alice logged in from 192.168.1.1
{"template": "<TIMESTAMP> User <USERNAME> logged in from <IP_ADDRESS>",
 "variables": {
   "TIMESTAMP": "2023-01-01 10:23:45",
   "USERNAME": "alice",
   "IP_ADDRESS": "192.168.1.1"
 }}

Example 2:
Log Line:
Error: failed to connect to database db42 on host 10.0.0.2
{"template": "Error: failed to connect to database <DB_NAME> on host <IP_ADDRESS>",
 "variables": {
   "DB_NAME": "db42",
   "IP_ADDRESS": "10.0.0.2"
 }}
"""


def main():
    # Sample logs
    logs = [
        "2024-03-12 12:43:22,934 - user john_doe logged in from 192.168.1.1",
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
        "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864",
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4",
        "17/06/09 20:10:40 INFO executor.CoarseGrainedExecutorBackend: Registered signal handlers for [TERM, HUP, INT]",
        "17/06/09 20:10:40 INFO spark.SecurityManager: Changing view acls to: yarn,curi",
        "Error: failed to connect to database worker64 on host 10.1.4.2"
    ]

    for log in logs:
        print(f"\n--- Processing log ---\n{log}")
        result = extract_template_and_variables_fewshot(log, FEW_SHOT_EXAMPLES)
        print("Parsed result:", json.dumps(result, indent=2))

        if "template" in result:
            embedding = get_embedding(normalize_template(result["template"]))
            print(f"Embedding length: {len(embedding)}")


if __name__ == "__main__":
    main()
