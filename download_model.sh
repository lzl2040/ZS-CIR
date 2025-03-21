# cd models
# for i in 1 2 3 4; do
#     wget https://huggingface.co/lmms-lab/llama3-llava-next-8b/resolve/main/model-0000$i-of-00004.safetensors
# done
# cd -
python load_llama3_hf.py
# rm models/*.safetensors