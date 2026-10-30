"""Despacho CUDA en procesos y entornos independientes por framework."""
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def worker(framework, action, payload, callback=None, _python=None):
    interpreter = Path(_python or f'/opt/{framework}/bin/python')
    if not interpreter.exists():
        raise ValueError('CUDA requiere iniciar Docker con compose.cuda.yaml. Consulta el README.')
    import numpy as np
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        request = directory/'request.json'
        request.write_text(json.dumps(dict(action=action, payload=payload)))
        env = dict(os.environ, NN2GUI_WORKER='1', PYTHONPATH='/app/src')
        result = None
        with (directory/'stderr').open('w+') as errors:
            process = subprocess.Popen([str(interpreter), '-m', 'proyecto.compute', str(request)],
                                       stdout=subprocess.PIPE, stderr=errors, text=True, env=env)
            try:
                for line in process.stdout:
                    if not line.startswith('NN2GUI:'):
                        continue
                    event = json.loads(line[7:])
                    if event['type'] == 'layer' and callback:
                        callback(event['name'], np.load(directory/event['file'], allow_pickle=False))
                    elif event['type'] == 'progress' and callback:
                        callback(event['value'])
                    elif event['type'] == 'result':
                        result = event['value']
                    elif event['type'] == 'error':
                        raise RuntimeError(event['value'])
                if process.wait() or result is None:
                    errors.seek(0)
                    raise RuntimeError('Falló el proceso CUDA: ' + errors.read()[-1500:])
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                process.stdout.close()
        return result


def validate_device(device):
    if device != 'cpu' and not re.fullmatch(r'cuda:\d+', device):
        raise ValueError('Dispositivo inválido; usa cpu o cuda:N.')


def probe(framework):
    try:
        return worker(framework, 'probe', {'framework': framework}), ''
    except (ValueError, RuntimeError, OSError) as exc:
        return [], str(exc)


def main():
    import sys
    import numpy as np
    request = Path(sys.argv[1])
    data = json.loads(request.read_text())
    action, args = data['action'], data['payload']
    def emit(kind, **values):
        print('NN2GUI:'+json.dumps(dict(type=kind, **values)), flush=True)
    def progress(value):
        emit('progress', value=value)
    def layer(name, values):
        path = request.parent/f'layer-{layer.count}.npy'
        layer.count += 1
        np.save(path, values, allow_pickle=False)
        emit('layer', name=name, file=path.name)
    layer.count = 0
    try:
        if action == 'probe':
            devices=[]
            if args['framework'] == 'torch':
                import torch
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        (torch.ones((2,2),device=f'cuda:{i}') @ torch.ones((2,2),device=f'cuda:{i}')).cpu()
                    devices.append([f'cuda:{i}', torch.cuda.get_device_name(i)])
            else:
                from proyecto.cnn import tensorflow
                tf=tensorflow()
                for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
                    with tf.device(f'/GPU:{i}'):
                        output=tf.linalg.matmul(tf.ones((2,2)), tf.ones((2,2)))
                    if 'GPU' not in output.device:
                        raise RuntimeError('TensorFlow no pudo ejecutar la operación en GPU.')
                    devices.append([f'cuda:{i}', tf.config.experimental.get_device_details(gpu).get('device_name', gpu.name)])
            result=devices
        elif action.startswith('mlp'):
            from proyecto.training import train, predict
            if action == 'mlp_train':
                import pandas as pd
                args['frame']=pd.read_json(io.StringIO(args['frame']),orient='split')
                result=train(**args, progress=progress)
            else:
                traced=args.pop('trace',False)
                result=predict(**args, on_layer=layer if traced else None).tolist()
        else:
            from proyecto.cnn import train_cnn, predict_cnn
            if action == 'cnn_train':
                result=train_cnn(**args, progress=progress)
            else:
                args['content']=base64.b64decode(args['content'])
                traced=args.pop('trace',False)
                result=predict_cnn(**args, on_layer=layer if traced else None).tolist()
        emit('result',value=result)
    except Exception as exc:
        emit('error', value=f'{type(exc).__name__}: {exc}')
        raise SystemExit(1)

if __name__ == '__main__':
    main()
