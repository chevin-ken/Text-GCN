from config import FLAGS
from model_factory import create_model
from utils import get_root_path, create_dir_if_not_exists, get_ts, save, sorted_nicely

from os.path import join
import torch


class Saver(object):
    def __init__(self):
        model_str = self.get_model_str()
        self.logdir = join(
            get_root_path(),
            'logs',
            '{}_{}'.format(model_str, get_ts()))
        create_dir_if_not_exists(self.logdir)
        self.model_info_f = self._open('model_info.txt')
        self._log_model_info()
        self._save_conf_code()
        print('Logging to {}'.format(self.logdir))

    def _log_model_info(self):
        s = get_model_info_as_str()
        c = get_model_info_as_command()
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.write(c)
        self.model_info_f.write('\n\n')

    def save_trained_model(self, trained_model, epoch=None):
        # Always save to the same filename, overwriting previous best model
        p = join(self.logdir, 'best_model.pt')
        torch.save(trained_model.state_dict(), p)
        epoch_str = f" (epoch {epoch})" if epoch is not None else ""
        print(f'Best model saved to {p}{epoch_str}')

    def load_trained_model(self, train_data):
        # Load the single best model file
        best_trained_model_path = join(self.logdir, 'best_model.pt')
        trained_model = create_model(train_data)
        trained_model.load_state_dict(
            torch.load(best_trained_model_path, map_location=FLAGS.device))
        trained_model.to(FLAGS.device)
        print(f'Loaded best model from {best_trained_model_path}')
        return trained_model

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(self.extract_config_code())
        p = join(self.logdir, 'FLAGS')
        print("in _save_conf_code")
        save({'FLAGS': FLAGS}, p, print_msg=False)

    def extract_config_code(self):
        with open(join(get_root_path(), 'config.py')) as f:
            return f.read()

    @staticmethod
    def get_model_str():
        li = []
        key_flags = [FLAGS.model.split(":")[0], FLAGS.dataset, "_".join([str(i) for i in FLAGS.tvt_ratio])]
        for f in key_flags:
            li.append(str(f))
        return '_'.join(li)

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

def get_model_info_as_str():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def get_model_info_as_command():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_root_path(), 'main.py'), '  '.join(rtn))