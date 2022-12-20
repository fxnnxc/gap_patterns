

export data='cifar10'
export seed=0 
export renew_last_layer=True
export freeze_pattern=True
export warmup_epochs=0

# model           train_e |eval_e |save_e
set1=('resnet18'  20      5       5)
set2=('resnet34'  20      5       5)
set3=('resnet50'  20      5       5)
set4=('resnet101' 20      5       5)
set5=('resnet152' 20      5       5)

pairs=(set1 set2 set3 set4 set5)
# pairs=(set4 set5)

for p in  ${pairs[@]}
do 
    echo $p
    declare -n pair=$p 
    pair=("${pair[@]}")
    encoder=${pair[0]}
    epochs=${pair[1]}
    eval_freq=${pair[2]}
    save_freq=${pair[3]}

    python src/run_encoder.py \
        --data $data \
        --encoder-name $encoder \
        --epochs $epochs \
        --eval-freq $eval_freq \
        --save-freq $save_freq \
        --renew-last-layer $renew_last_layer \
        --freeze-pattern $freeze_pattern \
        --warmup-epochs $warmup_epochs \
        --seed $seed

done