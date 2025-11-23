import json
import random
import argparse
from faker import Faker
from tqdm import tqdm

fake = Faker()

LABELS = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION",
]

# Noise injection functions
def spell_out_numbers(text):
    mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    return ''.join([mapping.get(c, c) if c.isdigit() else c for c in text])

def noisy_stt(text):
    text = text.lower()
    text = text.replace(".", " dot ")
    text = text.replace("@", " at ")
    text = text.replace("-", " ")
    text = spell_out_numbers(text)
    # Remove other punctuation
    text = "".join([c for c in text if c.isalnum() or c.isspace()])
    return " ".join(text.split())

def generate_example(id_num):
    # Templates for sentence generation
    templates = [
        "my name is {PERSON_NAME}",
        "call me at {PHONE}",
        "my email is {EMAIL}",
        "i live in {CITY}",
        "meet me at {LOCATION}",
        "my card is {CREDIT_CARD}",
        "the date is {DATE}",
        "contact {PERSON_NAME} at {EMAIL}",
        "is {PHONE} your number",
        "i was born in {CITY} on {DATE}",
        "send money to {CREDIT_CARD}",
        "visit {LOCATION} tomorrow",
        "hello this is {PERSON_NAME} speaking",
        "please update my email to {EMAIL}",
        "my new number is {PHONE}",
        "i am visiting {CITY} and {LOCATION}",
        "expiration date is {DATE}",
        "charge my card {CREDIT_CARD} please",
    ]
    
    template = random.choice(templates)
    
    entities = []
    text_parts = []
    
    # Split template by placeholders
    parts = template.split("{")
    
    current_text = parts[0]
    # Add noise to the static part
    current_text = noisy_stt(current_text)
    
    full_text = current_text
    
    for part in parts[1:]:
        label, rest = part.split("}")
        
        # Generate entity value
        if label == "PERSON_NAME":
            val = fake.name()
        elif label == "PHONE":
            val = fake.phone_number()
        elif label == "EMAIL":
            val = fake.email()
        elif label == "CITY":
            val = fake.city()
        elif label == "LOCATION":
            val = fake.address()
        elif label == "CREDIT_CARD":
            val = fake.credit_card_number()
        elif label == "DATE":
            val = fake.date()
        else:
            val = "unknown"
            
        # Apply noise to entity value
        val_noisy = noisy_stt(val)
        
        start = len(full_text)
        if full_text and not full_text.endswith(" "):
             full_text += " "
             start += 1
             
        full_text += val_noisy
        end = len(full_text)
        
        entities.append({
            "start": start,
            "end": end,
            "label": label
        })
        
        # Add the rest of the template
        rest_noisy = noisy_stt(rest)
        if rest_noisy:
            if not full_text.endswith(" "):
                full_text += " "
            full_text += rest_noisy

    return {
        "id": f"utt_{id_num:04d}",
        "text": full_text,
        "entities": entities
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    
    with open(args.output, "w", encoding="utf-8") as f:
        for i in tqdm(range(args.count)):
            example = generate_example(i)
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    main()
