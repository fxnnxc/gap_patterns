

export data='cifar10'
export seed=0 
export renew_last_layer=True


# model           train_e |eval_e |save_e
set1=('resnet18'  10      5       10)
set2=('resnet34'  10      5       10)
set3=('resnet50'  10      5       10)
set4=('resnet101' 10      5       10)
set5=('resnet152' 10      5       10)

pairs=(set1 set2 set3 set4 set5)

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
        --seed $seed

done