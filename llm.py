"""
Local LLM setup using Ollama.
Uses the locally installed model via Ollama API.
"""

import os
import requests
import json


class QwenLLM:
    """
    Wrapper class for LLM running via Ollama.
    """

    def __init__(self, model_path: str = "qwen2.5:7b", ollama_port: int = 11434):
        """
        Initialize the Ollama client.

        Args:
            model_path: Ollama model name (e.g., qwen2.5:7b, qwen2.5:7b-instruct).
            ollama_port: Port where Ollama is running (default: 11434).
        """
        self.model_name = model_path
        self.base_url = os.getenv("OLLAMA_HOST", f"http://localhost:{ollama_port}")

        print(f"Using Ollama model: {self.model_name}")

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("Ollama connection successful!")
            else:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama. Make sure it's running: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the model via Ollama.

        Args:
            prompt: The input prompt/question.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling (False = greedy decoding).

        Returns:
            The generated response text.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature if do_sample else 0,
                "top_p": top_p
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            raise Exception(f"Ollama error: {response.text}")

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response with a system prompt via Ollama.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user's question/input.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            The generated response text.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload
        )

        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "").strip()
        else:
            raise Exception(f"Ollama error: {response.text}")
