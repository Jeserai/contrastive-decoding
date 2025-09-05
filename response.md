1. What should you do if the two models have different tokenizers?

In most cases we want to avoid such cases because different tokenizers have different vocabulary sizes and token-to-ID mappings, which means we cannot calculate the logits subtraction with different vocabularies. It is not quite feasible to align two different tokenizers in general because it would require a bijection between them, which I believe doesn't exist in reality so far.

If forced to work with different tokenizers, potential workarounds include: (1) using a shared vocabulary subset, though this severely limits the effective vocabulary; (2) converting logits to probability distributions and attempting alignment, but this is computationally expensive and theoretically questionable; or (3) fine-tuning one model to use the other's tokenizer, which defeats the purpose of using pre-trained models. In practice, the best approach is to ensure both models use compatible tokenizers from the start.

2. Do you think contrastive decoding is used in practice?

I don't think contrastive decoding is widely or effectively used in practice. As an inference-level operation, contrastive decoding would greatly increase the computational overhead (we need two forward passes for each token we generate), let alone the memory usage to store an extra amateur model. I don't think the quality improvement brought by contrastive decoding can outweigh these defects, which makes it not quite economical.

Besides, the technique faces several practical challenges: (1) the need to maintain and serve two models increases infrastructure complexity; (2) the 2x inference cost makes it prohibitive for high-throughput applications; and (3) simpler alternatives like better prompting, fine-tuning, or RLHF often achieve similar quality improvements more efficiently.

However, contrastive decoding might be justified in specific scenarios: high-stakes applications where quality is paramount (legal, medical, safety-critical systems), research settings where computational cost is secondary to performance, or specialized domains where the amateur-expert pairing provides unique benefits. The technique is also valuable for research into model behavior and as inspiration for more efficient methods.

Overall, while contrastive decoding demonstrates interesting theoretical properties and can improve generation quality, its practical adoption is limited by computational costs. It's more likely to influence the development of more efficient techniques rather than see widespread direct deployment. 