# echo "nn_clf_ip_based" >> overhead.csv
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv
# python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo ip-based -output nn_clf_ip_based
# grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv

echo "nn_clf_outlier" >> overhead.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo heuristics-based-outlier -output nn_clf_outlier
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv

echo "nn_clf_quartile" >> overhead.csv
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv
python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 1 -data_retrain_duration_min 1 -data_eval_duration_min 1 -eval_period 8 -model_algo nn_clf -model_name nn_clf_1min -dd_algo heuristics-based-quartile -output nn_clf_quartile
grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> overhead.csv