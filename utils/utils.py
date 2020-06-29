import os
from torch import save


def save_model(model, model_name, checkpoint, experiment=None):
    if not os.path.exists('models/' + model_name):
            os.mkdir('models/' + model_name)

    file_path = 'models/' + model_name + '/' + model_name + '_' + str(checkpoint) + '.pt'
    model_name = model_name + '_' + str(checkpoint) + '.pt'
    save(model.state_dict(), file_path)

    if experiment is not None:
        with open(file_path, 'rb') as fp:
            experiment.log_model('PPO model', fp, file_name=model_name)


def interval_print(says, epoch):
    if epoch % 100 == 0:
        print(says)
