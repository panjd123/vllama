"""Example usage of vllama."""

import openai

# Configure client to use vllama
client = openai.OpenAI(
    base_url="http://localhost:33258/v1",
    api_key="not-needed"  # vllama doesn't require API key
)


def example_chat():
    """Example: Using a chat model."""
    print("\n=== Chat Example ===")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")


def example_chat_streaming():
    """Example: Streaming chat responses."""
    print("\n=== Streaming Chat Example ===")

    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "user", "content": "Write a short poem about coding."}
        ],
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def example_embeddings():
    """Example: Using an embedding model."""
    print("\n=== Embeddings Example ===")

    response = client.embeddings.create(
        model="Qwen/Qwen3-Embedding-0.6B",
        input=["Hello, world!", "How are you?"],
    )

    print(f"Number of embeddings: {len(response.data)}")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    print(f"First few values: {response.data[0].embedding[:5]}")


def example_list_models():
    """Example: List available models."""
    print("\n=== List Models Example ===")

    models = client.models.list()

    print(f"Available models: {len(models.data)}")
    for model in models.data:
        print(f"  - {model.id}")

def example_concurrent_requests(model_id: str):
    """Example: Making concurrent requests to the server."""
    import threading

    print("\n=== Concurrent Requests Example ===")

    def make_request(i):
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": f"What is {i} + {i}?"}
            ],
        )
        print(f"Response to request {i}: {response.choices[0].message.content}")

    threads = []
    for i in range(5):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    print("vllama Examples")
    print("=" * 50)
    print("\nMake sure vllama server is running:")
    print("  vllama serve")
    print("\nAnd that you have models cached in HF_HOME/hub")

    try:
        # List available models
        example_list_models()

        # Try different API endpoints
        # Note: These will only work if you have the corresponding models cached
        example_chat()
        example_chat_streaming()
        example_embeddings()

        example_concurrent_requests("Qwen/Qwen3-0.6B")
        example_concurrent_requests("Qwen/Qwen3-VL-2B-Instruct")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. vllama server is running (vllama serve)")
        print("2. You have models in HF_HOME/hub")
        print("3. The model names match your cached models")
