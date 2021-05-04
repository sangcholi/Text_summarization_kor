# Text_summarization_kor

한국어 문서 추출 모델

python3 train.py --gradient_clip_val 1.0 --max_epochs 1 --default_root_dir /KoBART-summarization/ --gpus 1 --batch_size 4 --num_workers 4 \
--checkpoint_path /checkpoint
