# echo "nn_clf_labeler" >> overhead3.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 0.5 -model_algo nn_clf -model_name nn_clf_1min -dd_algo heuristics-based-labeler -output nn_clf_labeler
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv

echo "nn_clf_psi" >> overhead3.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo population-stability-index -output nn_clf_psi
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv

# echo "nn_clf_psi_longer" >> overhead3.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo population-stability-index -output nn_clf_psi
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv

echo "nn_clf_noretrain" >> overhead3.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -no_retrain -output nn_clf_noretrain
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv

echo "nn_clf_ks" >> overhead3.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo kolmogorov-smirnov -output nn_clf_ks
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead3.csv