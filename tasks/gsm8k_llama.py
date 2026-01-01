"""GSM8K Llama task - Chain of Thought few-shot math evaluation."""
from __future__ import annotations

import re
from typing import Any


# Few-shot examples for chain-of-thought prompting
GSM8K_FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    },
]


def gsm8k_doc_to_text(doc: dict[str, Any]) -> str:
    """Convert GSM8K document to prompt text."""
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {doc['question']}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


def gsm8k_doc_to_target(doc: dict[str, Any]) -> str:
    """Extract target answer from GSM8K document."""
    answer = doc.get("answer", doc.get("target", ""))
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer


def gsm8k_extract_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response using regex filters."""
    # Try strict pattern first
    match = re.search(r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))", response)
    if match:
        return match.groups()[-1] or match.group(1)
    # Fall back to flexible pattern
    match = re.search(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    if match:
        return match.groups()[-1] or match.group(1)
    return response


def get_gsm8k_llama_config(TaskConfig):
    """Return the GSM8K Llama task configuration."""
    return TaskConfig(
        task="gsm8k_llama",
        dataset_path="gsm8k",
        dataset_name="main",
        doc_to_text=gsm8k_doc_to_text,
        doc_to_target=gsm8k_doc_to_target,
        extract_answer=gsm8k_extract_answer,
        fewshot_examples=GSM8K_FEWSHOT_EXAMPLES,
        stop_sequences=["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"],
        max_tokens=512,
    )
