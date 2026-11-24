"""Generate one document per prompt, base or adapter. Never contacts a RAG service."""
import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--limit',type=int);p.add_argument('--max-new-tokens',type=int,default=512);p.add_argument('--model-config',type=Path,default=ROOT/'configs/model.json');a=p.parse_args()
    if a.output.exists():raise SystemExit('Output exists. Choose a new filename to keep experimental runs separate.')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise SystemExit('Use one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0.')
    m=json.loads(a.model_config.read_text())
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok=AutoTokenizer.from_pretrained(m['model_id'],revision=m['revision'],trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(m['model_id'],revision=m['revision'],device_map={'':0},torch_dtype=dtype,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),attn_implementation='sdpa',trust_remote_code=False,use_safetensors=True)
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
    model.eval();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for i,line in enumerate(a.input.open()):
            if a.limit is not None and i>=a.limit:break
            row=json.loads(line)
            if [x['role'] for x in row['messages']]!=['system','user']:raise ValueError('Generation input must not contain reference assistant text.')
            inputs=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',**m.get('chat_template_kwargs',{})).to('cuda:0')
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            new=ids[0,inputs['input_ids'].shape[1]:]
            record={'example_id':row['example_id'],'group_id':row['group_id'],'source':row.get('source'),'generated_document':tok.decode(new,skip_special_tokens=True).strip(),'output_tokens':len(new),'duration_ms':round((time.perf_counter()-started)*1000,3),'model':m,'adapter':str(a.adapter) if a.adapter else None,'generation':{'do_sample':False,'max_new_tokens':a.max_new_tokens,'quantization':'nf4'},'evaluation':row.get('evaluation'),'rag_executed':False}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            print('Saved',row['example_id'],flush=True)
if __name__=='__main__':main()
