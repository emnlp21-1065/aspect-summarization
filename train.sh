python train_bart.py \
    --data_dir data/processed_data/yelp_big\
    --bert_model pretrained_model\
    --GPU_index "2"\
    --train_batch_size 4\
    --num_train_epochs 10\
    --learning_rate 1e-4\
    --print_every 100\
    --output_dir output

