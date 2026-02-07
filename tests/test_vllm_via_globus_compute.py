from globus_compute_sdk import Executor

endpoint_id = "8d978809-eec4-413d-bbd4-b099e488100a"


def test_vllm_inference(messages):
    """Call vLLM server via HTTP"""
    import os

    import requests

    vllm_url = os.environ.get("VLLM_SERVER_URL")

    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 100,
        },
        timeout=60,
    )
    return response.json()


if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "What is quantum computing in one sentence?"}]

    with Executor(endpoint_id=endpoint_id) as gce:
        future = gce.submit(test_vllm_inference, test_messages)
        print("Task submitted, waiting for vLLM inference...")

        result = future.result(timeout=120)
        print("\n✅ vLLM INFERENCE SUCCESS!")
        print(f"Response: {result['choices'][0]['message']['content']}")
