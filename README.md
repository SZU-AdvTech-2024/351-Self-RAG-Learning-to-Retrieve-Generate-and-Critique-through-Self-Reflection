## 使用教程

### 1.首先运行以下命令训练批判模型

```
python train_critic_model.py --model_name_or_path path_to_base_model 
--data_path path_to_train_data_file  --bf16 True 
--output_dir path_to_critic_model  --num_train_epochs 3 
--per_device_train_batch_size 1  --per_device_eval_batch_size 1 
--gradient_accumulation_steps 8  --evaluation_strategy "no" 
--save_strategy "steps"  --save_steps 300 
--save_total_limit 1  --learning_rate 2e-5 
--weight_decay 0.  --warmup_ratio 0.01 
--lr_scheduler_type "cosine"  --logging_steps 10
```

### 2.然后运行以下命令测试结果

```
python main.py  --critic_model_path path_to_critic_model  --generate_model_path path_to_generate_model 
--retrieve_model_path path_to_retrieve_model  --retrieve_mode "with_retrieve_context" or "without_retrieve_context" 
--input_file test_data  --dataset_name dataset_name 
--max_critic max_critic_times  --passages path_to_passages 
--passages_embeddings path_to_passages_embeddings  --n_docs_number_of_documents_to_retrieve_per_questions 
```
### 3.检索文档及检索文档的embeddings下载链接

```
https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```

### 4.测试数据下载链接

```
https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb/view?usp=share_link
```