# echo "nn_clf_kl" >> overhead4.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo kullback-leibler -output nn_clf_kl
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv

# echo "nn_clf_js" >> overhead4.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo jensen-shannon -output nn_clf_js
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv

echo "nn_clf_labeler" >> overhead4.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1.5 -eval_period 0.5 -model_algo nn_clf -model_name nn_clf_1min -dd_algo heuristics-based-labeler -output nn_clf_labeler
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv

echo "nn_clf_js" >> overhead4.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1.5 -eval_period 0.5 -model_algo nn_clf -model_name nn_clf_1min -dd_algo jensen-shannon -output nn_clf_js
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead4.csv