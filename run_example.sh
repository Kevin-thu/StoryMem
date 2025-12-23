export PYTHONNOUSERSITE=1
DISTRIBUTED_ARGS="--nproc_per_node=8 --master_port=9999"
OUTPUT_PATH=results
mkdir -p $OUTPUT_PATH

torchrun $DISTRIBUTED_ARGS  pipeline.py \
--story_script_path ./story/daiyu.json \
--t2v_model_path ./models/Wan2.2-T2V-A14B \
--i2v_model_path ./models/Wan2.2-I2V-A14B \
--lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B \
--size "832*480" --max_memory_size 10 \
--output_dir $OUTPUT_PATH \
--dit_fsdp --t5_fsdp --ulysses_size 8 --offload_model --lora_rank 128 --mi2v --t2v_first_shot

torchrun $DISTRIBUTED_ARGS  pipeline.py \
--story_script_path ./story/daiyu.json \
--t2v_model_path ./models/Wan2.2-T2V-A14B \
--i2v_model_path ./models/Wan2.2-I2V-A14B \
--lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B \
--size "832*480" --max_memory_size 10 \
--output_dir $OUTPUT_PATH \
--dit_fsdp --t5_fsdp --ulysses_size 8 --offload_model --lora_rank 128 --mi2v