"""Build completion-only labels without silent truncation. Used by trainer and checks."""
import json
from pathlib import Path


def encode_example(tokenizer,row,max_length,chat_template_kwargs=None):
    messages=row['messages']
    options=chat_template_kwargs or {}
    prompt=tokenizer.apply_chat_template(messages[:2],tokenize=True,add_generation_prompt=True,**options)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,**options)
    if full[:len(prompt)]!=prompt:
        raise ValueError('Tokenizer template prefix mismatch: '+row['example_id'])
    if len(full)>max_length:
        raise ValueError(f'{row["example_id"]}: {len(full)} tokens exceeds {max_length}; increase the limit or curate the example, do not silently drop its target.')
    labels=[-100]*len(prompt)+full[len(prompt):]
    if not any(x!=-100 for x in labels):raise ValueError('No completion labels')
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}


def load_encoded(path,tokenizer,max_length,limit=None,chat_template_kwargs=None):
    result=[]
    for line in Path(path).open():
        row=json.loads(line)
        result.append(encode_example(tokenizer,row,max_length,chat_template_kwargs))
        if limit is not None and len(result)>=limit:break
    return result
