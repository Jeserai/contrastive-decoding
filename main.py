import transformers as tr
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    amateur.to("cuda")
    expert.to("cuda")
    device = expert.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids
    past_key_values_amateur = None
    past_key_values_expert = None

    for _ in range(max_tokens):
        # Use the last token for the next step prediction
        current_token_ids = generated_ids[:, -1:]

        # 3. Get logits from both models using the KV cache
        with torch.no_grad():
            # Get logits from the amateur model
            amateur_outputs = amateur(
                current_token_ids, past_key_values=past_key_values_amateur, use_cache=True
            )
            logits_amateur = amateur_outputs.logits[:, -1, :]
            past_key_values_amateur = amateur_outputs.past_key_values

            # Get logits from the expert model
            expert_outputs = expert(
                current_token_ids, past_key_values=past_key_values_expert, use_cache=True
            )
            logits_expert = expert_outputs.logits[:, -1, :]
            past_key_values_expert = expert_outputs.past_key_values

        # 4. Combine logits using the contrastive decoding formula
        # We work with logits to maintain numerical stability
        contrastive_logits = 1.5 * logits_expert - 0.5 * logits_amateur

        # 5. Select the next token using greedy search (argmax)
        next_token_id = torch.argmax(contrastive_logits, dim=-1).unsqueeze(-1)

        # 6. Append the new token and check for EOS
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            print("EOS token reached.")
            break
            
    # 7. Decode the final generated sequence
    # We skip the special tokens and the original prompt
    return tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path, torch_dtype=torch_dtype, device_map="auto")
    expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch_dtype, device_map="auto")

    generated_text = contrastive_generation(
        amateur=amateur_model,
        expert=expert_model,
        prompt=prompt,
        max_tokens=100
    )

    print("Prompt:")
    print(user_message)
    print("\nGenerated Docstring:")
    print(generated_text)

if __name__ == "__main__":
    main()