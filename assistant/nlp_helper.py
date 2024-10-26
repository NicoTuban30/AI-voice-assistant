# assistant/nlp_helper.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Set a maximum token limit for conversation history
MAX_HISTORY_LENGTH = 200  # Adjust based on context needs and model's capability
chat_history_ids = None


def generate_response(user_input):
    global chat_history_ids

    # Encode the user input and add EOS token
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )

    # Concatenate new input with chat history, if available
    if chat_history_ids is not None:
        # Truncate history to maintain token limit
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids.size(-1) > MAX_HISTORY_LENGTH:
            chat_history_ids = chat_history_ids[
                :, -MAX_HISTORY_LENGTH:
            ]  # Trim history from the start if too long
    else:
        chat_history_ids = new_input_ids

    # Generate attention mask for input length
    attention_mask = torch.ones(chat_history_ids.shape, dtype=torch.long)

    # Generate a response using the model
    response_ids = model.generate(
        chat_history_ids,
        max_length=chat_history_ids.shape[-1]
        + 50,  # Allow a buffer for response length
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Control randomness with temperature
    )

    # Extract the response part of the generated output
    response = tokenizer.decode(
        response_ids[:, chat_history_ids.shape[-1] :][0], skip_special_tokens=True
    )

    # Update conversation history with the response
    chat_history_ids = torch.cat(
        [chat_history_ids, response_ids[:, chat_history_ids.shape[-1] :]], dim=-1
    )

    return response
