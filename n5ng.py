#!/usr/bin/env python3

import argparse
import gzip
import io
import z5py
import numpy as np
from flask import Flask, jsonify, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_scales(dataset_name, scales, encoding='raw', base_res=np.array([1.0,1.0,1.0])):
    def get_scale_for_dataset(dataset, scale, base_res):
        if 'resolution' in dataset.attrs:
            resolution = dataset.attrs['resolution']
        elif 'downsamplingFactors' in dataset.attrs:
            # The FAFB n5 stack reports downsampling, not absolute resolution
            resolution = (base_res * np.asarray(dataset.attrs['downsamplingFactors'])).tolist()
        else:
            resolution = (base_res*2**scale).tolist()
        return {
                    'chunk_sizes': [list(reversed(dataset.chunks))],
                    'resolution': resolution,
                    'size': list(reversed(dataset.shape)),
                    'key': str(scale),
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
                this_scale = scale_info.append(get_scale_for_dataset(dataset, scale, base_res))
            except Exception as exc:
                print(exc)
    else:
        dataset = app.config['n5file'][dataset_name]
        # No scale pyramid for this dataset
        scale_info = [ get_scale_for_dataset(dataset, 1.0, base_res) ]       
    return scale_info

@app.route('/<path:dataset_name>/info')
def dataset_info(dataset_name):
    dataset = app.config['n5file'][dataset_name]
    info = {
        'data_type' : 'uint8',
        'type': 'image',
        'num_channels' : 1,
        'scales' : get_scales(dataset_name, scales=list(range(0,8)), base_res=np.array([4.0, 4.0, 40.0]))
        
    }
    return jsonify(info)

# Implement the neuroglancer precomputed filename structure for URL requests
@app.route('/<path:dataset_name>/<int:scale>/<int:x1>-<int:x2>_<int:y1>-<int:y2>_<int:z1>-<int:z2>')
def get_data(dataset_name, scale, x1, x2, y1, y2, z1, z2):
    # TODO: Enforce a data size limit
    dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
    dataset = app.config['n5file'][dataset_name_with_scale]
    # z5py 
    data = dataset[z1:z2,y1:y2,x1:x2]
    # Neuroglancer expects an x,y,z array in Fortram order (e.g., z,y,x in C =)
    response = Response(data.tobytes(order='C'), mimetype='application/octet-stream')

    accept_encoding = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower() or \
           'Content-Encoding' in response.headers:
            return response

    gzip_buffer = io.BytesIO()
    gzip_file = gzip.GzipFile(mode='wb', compresslevel=5, fileobj=gzip_buffer)
    gzip_file.write(response.data)
    gzip_file.close()
    response.data = gzip_buffer.getvalue()
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.data)

    return response



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