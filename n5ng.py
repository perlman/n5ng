#!/usr/bin/env python3

import argparse
import z5py
import numpy as np
from flask import Flask, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_scales(dataset_name, scales, encoding='raw'):
    def get_scale_for_dataset(dataset):
        return {
                    'chunk_sizes': [dataset.chunks],
                    'resolution': dataset.attrs.get('resolution', [1.0,1.0,1.0]),
                    'size': dataset.shape,
                    'key': '0',
                    'encoding': encoding,
                    'voxel_offset': dataset.attrs.get('offset', [0,0,0]),
                }
                
    if  scales:
        # Assumed scale pyramid with the convention dataset/sN, where N is the scale level
        scale_info = []
        for scale in scales:
            try:
                dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
                dataset = app.config['n5file'][dataset_name_with_scale]
                this_scale = scale_info.append(get_scale_for_dataset(dataset))
            except Exception as exc:
                print(exc)
                pass
    else:
        dataset = app.config['n5file'][dataset_name]
        # No scale pyramid for this dataset
        scale_info = [ get_scale_for_dataset(dataset) ]       
    return scale_info

@app.route('/<path:dataset_name>/info')
def dataset_info(dataset_name):
    dataset = app.config['n5file'][dataset_name]
    info = {
        'data_type' : 'uint8',
        'type': 'image',
        'num_channels' : 1,
        'scales' : get_scales(dataset_name, scales=list(range(0,10)))
        
    }
    return jsonify(info)

# Implement the neuroglancer precomputed filename structure for URL requests
@app.route('/<path:dataset_name>/<int:scale>/<int:x1>-<int:x2>_<int:y1>-<int:y2>_<int:z1>-<int:z2>')
def get_data(dataset_name, scale, x1, x2, y1, y2, z1, z2):
    # TODO: Enforce a data size limit
    dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
    dataset = app.config['n5file'][dataset_name_with_scale]
    data = dataset[x1:x2,y1:y2,z1:z2]
    return Response(data.tobytes(order='F'), mimetype='application/octet-stream')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='n5 file to share', default='sample.n5')
    args = parser.parse_args()

    n5f = z5py.file.N5File(args.filename, mode='r')

    # Start flask
    app.debug = True
    app.config['n5file'] = n5f
    app.run(host='0.0.0.0')

if __name__ == '__main__':
    main()