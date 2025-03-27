echo "eval orginal"
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines=1 --num_processes 1 --machine_rank 0 retrieval.py \
                                         --llava_llama3 \
                                         --lora_path '/mnt/output_zuo/ZS-CIR/e5v-8b-4bit-ke/checkpoint-400' \
                                         --file-path '/mnt/output_zuo/ZS-CIR/results' \
                                         --prompt_type 'org'
echo "eval knowledge enhancments"
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines=1 --num_processes 1 --machine_rank 0 retrieval.py \
                                         --llava_llama3 \
                                         --lora_path '/mnt/output_zuo/ZS-CIR/e5v-8b-4bit-ke/checkpoint-400' \
                                         --file-path '/mnt/output_zuo/ZS-CIR/results' \
                                         --prompt_type 'ke'
echo "eval chain of thought"
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines=1 --num_processes 1 --machine_rank 0 retrieval.py \
                                         --llava_llama3 \
                                         --lora_path '/mnt/output_zuo/ZS-CIR/e5v-8b-4bit-ke/checkpoint-400' \
                                         --file-path '/mnt/output_zuo/ZS-CIR/results' \
                                         --prompt_type 'cot'