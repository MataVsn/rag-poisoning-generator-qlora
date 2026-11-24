"""Single-GPU completion-only QLoRA. Run on the rented host after preflight."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/qlora_4b.json');p.add_argument('--dataset',choices=['mixed','synthetic_ru','poisonedrag']);p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--benchmark-only',action='store_true');a=p.parse_args()
    if a.benchmark_only and a.smoke_steps<=0:p.error('--benchmark-only requires --smoke-steps > 0')
    import torch
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,DataCollatorForSeq2Seq,Trainer,TrainingArguments,EarlyStoppingCallback,TrainerCallback,set_seed
    from tokenize_data import load_encoded
    if not torch.cuda.is_available():raise SystemExit('CUDA required; run bootstrap/preflight on your rented GPU.')
    if torch.cuda.device_count()!=1:raise SystemExit('Expose ONE GPU: CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py ...')
    cfg=json.loads(a.config.read_text());model_cfg=json.loads((ROOT/cfg.get('model_config','configs/model.json')).read_text())
    dataset=a.dataset or cfg['dataset'];set_seed(cfg['seed'])
    target=ROOT/cfg['output_dir']
    if a.smoke_steps:target=target/'smoke'
    if (target/'adapter').exists():raise SystemExit('Existing adapter output; choose a new output_dir rather than overwrite.')
    target.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(model_cfg['model_id'],revision=model_cfg['revision'],trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    train_path=ROOT/'data/sft'/dataset/'train.jsonl';val_path=ROOT/'data/sft'/dataset/'validation.jsonl'
    train=load_encoded(train_path,tokenizer,cfg['max_length'],limit=64 if a.smoke_steps else None,chat_template_kwargs=model_cfg.get('chat_template_kwargs'))
    val=load_encoded(val_path,tokenizer,cfg['max_length'],limit=8 if a.smoke_steps else None,chat_template_kwargs=model_cfg.get('chat_template_kwargs'))
    bf16=torch.cuda.is_bf16_supported();dtype=torch.bfloat16 if bf16 else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(model_cfg['model_id'],revision=model_cfg['revision'],quantization_config=quant,torch_dtype=dtype,device_map={'':0},attn_implementation='sdpa',trust_remote_code=False,use_safetensors=True)
    model.config.use_cache=False
    checkpointing=cfg.get('gradient_checkpointing',True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=checkpointing,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=cfg['target_modules'],bias='none',task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    args=TrainingArguments(output_dir=str(target/'checkpoints'),num_train_epochs=cfg['epochs'],max_steps=a.smoke_steps if a.smoke_steps else -1,per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=cfg.get('eval_batch_size',1),gradient_accumulation_steps=cfg['gradient_accumulation_steps'],learning_rate=cfg['learning_rate'],warmup_ratio=.05,lr_scheduler_type='cosine',optim=cfg.get('optim','adamw_torch'),bf16=bf16,fp16=not bf16,gradient_checkpointing=checkpointing,gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1 if a.smoke_steps else 10,eval_strategy='no' if a.smoke_steps else 'steps',eval_steps=cfg['eval_steps'],save_strategy='no' if a.smoke_steps else 'steps',save_steps=cfg['save_steps'],save_total_limit=2,load_best_model_at_end=not bool(a.smoke_steps),metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0,save_safetensors=True)
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100,pad_to_multiple_of=8)
    callbacks=[]
    if not a.smoke_steps and cfg.get('early_stopping_patience'):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience']))
    class WallTimeLimit(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic()
        def on_step_end(self,args,state,control,**kwargs):
            if time.monotonic()-self.started>=cfg['max_training_seconds']:
                control.should_training_stop=True
                control.should_evaluate=True
                control.should_save=True
    if not a.smoke_steps and cfg.get('max_training_seconds'):
        callbacks.append(WallTimeLimit())
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=collator,processing_class=tokenizer,callbacks=callbacks)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    result=trainer.train();metrics=trainer.evaluate()
    if not all(math.isfinite(float(v)) for v in [metrics['eval_loss'],result.metrics['train_loss']]):raise RuntimeError('Non-finite loss: do not save or accept this run')
    adapter=target/'adapter'
    if not a.benchmark_only:
        trainer.save_model(str(adapter));tokenizer.save_pretrained(adapter)
    metrics.update(result.metrics)
    (target/'training_metrics.json').write_text(json.dumps(metrics,indent=2,default=str))
    (target/'trainer_history.json').write_text(json.dumps(trainer.state.log_history,indent=2,default=str))
    manifest={'model':dict(model_cfg,weights_downloaded=True,training_verified_on_rented_gpu=True),'config':cfg,'dataset':dataset,'train_examples':len(train),'validation_examples':len(val),'smoke_only':bool(a.smoke_steps),'benchmark_only':a.benchmark_only,'adapter_saved':not a.benchmark_only,'test_split_used':False,'runtime_seconds':time.perf_counter()-started,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'device':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,'bnb_cuda_override':os.environ.get('BNB_CUDA_VERSION'),'python':platform.python_version(),'packages':{k:version(k) for k in ['torch','transformers','peft','accelerate','bitsandbytes']},'input_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [train_path,val_path]},'best_validation_loss':trainer.state.best_metric,'best_checkpoint':trainer.state.best_model_checkpoint,'optimizer_steps':trainer.state.global_step,'trainable_parameters':model.get_nb_trainable_parameters()[0],'all_parameters':model.get_nb_trainable_parameters()[1],'metric_limit':'Validation loss is not RAG attack success.'}
    (target/'training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Benchmark completed without saving adapter' if a.benchmark_only else 'Adapter saved: '+str(adapter))

if __name__=='__main__':main()
