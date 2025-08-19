import openai
import prompts_ as prompts
import utils_ as utils

import os
import json

import torch
from unsloth import FastLanguageModel

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

# Class for handling program repair using language models
class Repair:
    # Initialize the Repair class with model details
    def __init__(self, model_name, is_huggingface, compiler = None):
        print("model_name:", model_name)
        print("is_huggingface:", is_huggingface)
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        self.user_prompt = prompts.user_message_nus_repair_basic
        self.is_huggingface = is_huggingface
        self.compiler = compiler
        self.distance = utils.Distance()
        if self.is_huggingface:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, 
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",
            )
            FastLanguageModel.for_inference(self.model)
        else:
            with open("YOUR_openai.txt") as f:#HERE
                content = f.read().strip()
                openai_key = content.splitlines()[2]
                openai.api_key = openai_key

    # Extract fixed code from the generated text
    def extract_fixed_code(self, text):
        start_tag = "[FIXED]"
        end_tag = "[/FIXED]"
        
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)
        
        extracted_text = text[start_index:end_index].strip()
        
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.startswith("python"):
            extracted_text = extracted_text[6:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        
        return extracted_text

    # Save the transcript to a JSON file at "project_part1_transcripts/transcript.json". This file contains all prompts and LLM responses which can be used for debugging.
    def save_transcript_to_json(self, transcript):
        os.makedirs("project_part1_transcripts", exist_ok=True)
        file_path = os.path.join("project_part1_transcripts", "transcript.json")
        
        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        
        # Append new transcript data
        existing_data.extend(transcript)
        
        # Write back to the file
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    # Call the OpenAI language model
    def call_llm_openai(self, system_prompt_formatted, user_prompt_formatted):
        responses = []
        for _ in range(3):  # Generate 3 repair candidates
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_formatted},
                    {"role": "user", "content": user_prompt_formatted}
                ],
                temperature=0.7  # Set temperature to 0.7
            )
            responses.append(response.choices[0].message.content)
        return responses

    # Call the Hugging Face language model
    def call_llm_huggingface(self, system_prompt_formatted, user_prompt_formatted):
        responses = []
        for _ in range(3):  # Generate 3 repair candidates
            prompt_string = f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""        
            inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
            outputs = self.model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_new_tokens=2048, 
                use_cache=True,
                do_sample=True,  # Enable sampling
                temperature=0.7  # Set temperature to 0.7
            )
        
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            responses.append(response_text)
        return responses

    # Call the appropriate language model based on configuration
    def call_llm(self, problem_data, buggy_program):
        system_prompt_formatted = self.system_prompt
        user_prompt_formatted = self.user_prompt.format(problem_data=problem_data, buggy_program=buggy_program)
        
        transcript = []
        
        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(system_prompt_formatted, user_prompt_formatted)
        else:
            generated_response = self.call_llm_openai(system_prompt_formatted, user_prompt_formatted)
        
        transcript.append({
            "input_prompt": system_prompt_formatted + user_prompt_formatted,
            "output": generated_response
        })
            
        self.save_transcript_to_json(transcript)
        
        return generated_response

    # Generate a repair for the given problem and program
    def generate_repair(self, problem_data, buggy_program, testcases):        
        generated_responses = self.call_llm(problem_data, buggy_program)
        best_repair = None
        best_distance = float('inf')
    
        for response in generated_responses:
            fixed_code = self.extract_fixed_code(response)
            if fixed_code:
                all_correct, _ = self.compiler.run_program_with_testcases(fixed_code, testcases)
                if all_correct:
                    distance = self.distance.get_edit_distance(fixed_code, buggy_program)
                    if distance < best_distance:
                        best_repair = fixed_code
                        best_distance = distance
    
        return best_repair if best_repair else ""
