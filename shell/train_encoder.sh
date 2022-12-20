

# export data='cifar10'
# export encoder='resnet50'
# export epochs=10
# export eval_freq=3
# export save_freq=10
# export seed=0 
# export renew_last_layer=True

python src/run_encoder.py \
    --data $data \
    --encoder-name $encoder \
    --epochs $epochs \
    --eval-freq $eval_freq \
    --save-freq $save_freq \
    --renew-last-layer $renew_last_layer \
    --seed $seed
