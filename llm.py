"""
Local LLM setup using Qwen 2.5 7B.
Runs locally with automatic device detection.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device_config():
    """Detect the best available device and return appropriate config."""
    if torch.cuda.is_available():
        try:
            torch.tensor([1.0]).cuda()
            return {"device_map": "auto", "torch_dtype": torch.float16}
        except Exception:
            pass
    # CPU fallback
    return {"device_map": "cpu", "torch_dtype": torch.float32}


class QwenLLM:
    """
    Wrapper class for Qwen 2.5 7B model running locally.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the Qwen model.

        Args:
            model_path: HuggingFace model ID or local path to the model.
                       If you have the model downloaded locally, use that path.
        """
        print(f"Loading Qwen model from {model_path}...")

        device_config = get_device_config()
        print(f"Using device config: {device_config}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **device_config
        )

        self.model.eval()
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt/question.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling (False = greedy decoding).

        Returns:
            The generated response text.
        """
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response with a system prompt.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user's question/input.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            The generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return response.strip()
