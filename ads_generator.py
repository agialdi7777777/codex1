import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


MODEL_NAME = "google/flan-t5-base"

def load_model(model_name: str):
    """Load a Hugging Face model and return a text-generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)


def generate_ad(topic: str) -> str:
    """Generate an SMS ad for the given topic."""
    text_gen = load_model(MODEL_NAME)
    hf_llm = HuggingFacePipeline(pipeline=text_gen)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short and catchy SMS advertisement about {topic}."
    )
    chain = LLMChain(llm=hf_llm, prompt=prompt)
    return chain.run(topic=topic).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate SMS advertisement text")
    parser.add_argument("topic", help="Product or topic of the advertisement")
    args = parser.parse_args()
    ad_text = generate_ad(args.topic)
    print(ad_text)


if __name__ == "__main__":
    main()
