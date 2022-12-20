

export data='cifar10'
export encoder='resnet50'
export model='simple'
export cnn_dim=2048
export epochs=10
export eval_freq=3
export seed=0 

python src/run_gap.py \
    --data $data \
    --encoder-name $encoder \
    --model-name $model \
    --cnn-dim $cnn_dim \
    --epochs $epochs \
    --eval-freq $eval_freq \
    --seed $seed
