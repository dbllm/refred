# conformal prediction.
from cllm.reg.io import load_std_data
from cllm.io import split_data
from pathlib import Path
from cllm.reg.basecp import BaseRegCP, ConformalScore2Norm
from sklearn.svm import SVR

def predict(output_name):
    output_name = Path('./data_out/std/cp_predict/') / '{}.csv'.format(output_name)
    
    pdf_data = load_std_data('aug')
    train_data, calibrate_data, test_data = split_data(pdf_data, percentage=[0.4, 0.3, 0.3])

    model = SVR()
    cpmodel = BaseRegCP(model, score=ConformalScore2Norm())

    cpmodel.train(train_data)
    cpmodel.calibrate(calibrate_data, 0.3)
    res = cpmodel.test(test_data)
    print(res)
    # save test data.
    print(len(test_data))
    # get prediction sets for each 
    predictions = cpmodel.test(test_data, return_pred=True)
    # for _, _row in test_data.iterrows():
    test_data['pred'] = predictions.tolist()
    save_data = test_data.drop(columns=['dist_arr', 'emb', 'gt_emb'])
    output_name.parent.mkdir(parents=True, exist_ok=True)
    save_data.to_csv(output_name)


if __name__ == '__main__':
    predict('testrun_1')