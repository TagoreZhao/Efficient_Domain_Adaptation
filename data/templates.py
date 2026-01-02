
pubmed_input_template = (
    "Below is an instruction that describes a task related to HealthCare, "
    "paired with detailed input from a scientific article.\n\n"
    "Instruction: As an expert doctor in clinical science and medical knowledge, "
    "can you tell me if the following statement is correct? Answer yes, no, or maybe.\n\n"
    "Input:\n"
    "Contexts:\n{contexts}\n\n"
    "Section Labels: {labels}\n\n"
    "MeSH Terms: {meshes}\n\n"
    "Question: {question}\n\n"
    "Response: The answer is"
)
pubmed_target_template = " {ground_truth}"

mednli_input_template = (
    "Below is an instruction for analyzing clinical information.\n\n"
    "Instruction:\n"
    "You are a licensed physician tasked with determining the logical relationship between two clinical statements based on a patient's past medical history.\n"
    "Analyze carefully whether the second statement (hypothesis) can be inferred from the first statement (premise).\n"
    "Base your reasoning strictly on clinical knowledge, without making unwarranted assumptions.\n"
    "Your answer must be one of three options:\n"
    "- 'entailment' if the hypothesis must be true given the premise,\n"
    "- 'contradiction' if the hypothesis must be false given the premise,\n"
    "- 'neutral' if the hypothesis could be true or false without certainty.\n"
    "Fully consider all three possibilities and reason carefully before giving your final answer. Answer entailment, contradiction, or neutral.\n\n"
    "Input:\nPremise: '{sentence1}'\nHypothesis: '{sentence2}'\n\n"
    "Final Answer: Their relationship is"
)
mednli_target_template = " {gold_label}"

hqs_input_template = (
        "Below is an instruction that describes a task related to HealthCare, paired with further context. "
        "Write a response that appropriately completes the request.\n\n"
        "Instruction: Summarize the following consumer health question by generating a condensed version that retains all critical information necessary to find correct and complete answers. "
        "Focus on preserving key entities (e.g., conditions, treatments, tests) and the main intent of the question, while omitting unnecessary peripheral details. "
        "Write the summary fluently in natural language and avoid simply shortening without ensuring information completeness.\n\n"
        "Please provide the shortened version directly.\n\n"
        "Input: {input_question}\n\nShortened Question:"
)

hqs_target_template = " {sum}"

casehold_input_template = (
    "Below is an instruction that describes a task related to making legal decisions based on a citing prompt. "
    "Please read the citing prompt, and decide which holding statement best corresponds to it.\n\n"
    "Instruction:\n"
    "You are given a citing passage from a legal decision and five potential holding statements.\n"
    "Your task is to carefully read the citing prompt and select the holding statement that best corresponds to it.\n"
    "Focus on the precise legal principle, factual framing, and implications described.\n"
    "Base your decision purely on the logical match between the citing prompt and the holding, avoiding irrelevant or merely similar content.\n\n"
    "Your answer should be 0, 1, 2, 3, or 4.\n\n"
    "Input: citing_prompt: {citing_prompt}, 0: {holding_0}, 1: {holding_1}, 2: {holding_2}, 3: {holding_3}, 4: {holding_4}\n\n"
    "Response: The answer is"
)
casehold_target_template = " {label}"

billsum_input_template = (
        "Below is an instruction for summarizing a legislative bill.\n\n"
        "Instruction:\n"
        "Summarize the following legislative text by clearly identifying and explaining the *major actions, purposes, and effects* of the bill. "
        "Focus on what the bill aims to achieve rather than technical details or administrative changes. "
        "Paraphrase the information in a simple and accessible way as if writing for policymakers and the public. "
        "Avoid copying long passages or citing subsection numbers unless necessary for understanding.\n\n"
        "Input:\n"
        "Bill Title: {title}\n\n"
        "{input_text}\n\n"
        "Response:"
    )
billsum_target_template = "{summary}"

contractNLI_input_template = (
        "Below is an instruction describing a task related to legal contract analysis.\n\n"
        "Instruction:\n"
        "You are given a pair consisting of a Premise and a Hypothesis, both drawn from a formal legal contract.\n"
        "Carefully read the Premise and determine whether the Hypothesis:\n"
        " - (Entailment) Must be true if the Premise is true,\n"
        " - (Contradiction) Is directly refuted by the Premise,\n"
        " - (Neutral) Is neither clearly entailed nor contradicted by the Premise.\n\n"
        "You must think carefully and reason step-by-step, considering the precise meaning of the contract language.\n"
        "Your final answer must be one of exactly 'entailment', 'contradiction', or 'neutral'.\n\n"
        "Premise: {sentence1}\n"
        "Hypothesis: {sentence2}\n\n"
        "Response:\n"
        "Final Answer:"
    )
contractNLI_target_template = "{summary}"