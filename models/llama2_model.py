from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Llama-2-7B-chat-GGML"  # Ajusta según el modelo que desees
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
