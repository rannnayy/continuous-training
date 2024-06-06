echo "nurd" >> overhead6.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead6.csv
python3 model_reweight.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_name nurd -output nurd
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead6.csv

# echo "nn_clf_oracle" >> overhead6.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead6.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -oracle -output nn_clf_oracle
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead6.csv
