import numpy as np
import time
import ctypes
import cv2 as cv
import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as Executor

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--nproc', type=int, default=4)
args = parser.parse_args()

nvp = ctypes.CDLL('./NVP3000_DLL_MMX.DLL')
nvp.NVP3000_MJPEG.argtypes = (np.ctypeslib.ndpointer(
                                dtype=ctypes.c_ubyte,
                                flags='C_CONTIGUOUS'),
                              np.ctypeslib.ndpointer(
                                dtype=ctypes.c_ubyte,
                                flags='C_CONTIGUOUS'),
                              ctypes.c_ubyte,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int)
nvp.NVP3000_MJPEG.restype = ctypes.c_int

class Rect:
    def __init__(self, l, t, r, b):
        self.left = l
        self.top = t
        self.right = r
        self.bottom = b

def proc_file(file):
    path = Path(file)
    if not path.is_file():
        print(f'Failed to open file {file}')
        raise IOError(message=f'Failed to open file {file}', filename=file, errno=22)

    def read_header(buf):
        header = buf.read(16)
        
        if int.from_bytes(header[0:2], 'big') != 0xaaaa: return

        tmp_word = int.from_bytes(header[2:4], 'big')
        sector = tmp_word >> 7
        packet = tmp_word & 0x007f
        
        if not packet:
            jpeg_size = 512*sector - 32
        else:
            jpeg_size = 512*(sector-1) + (packet-1)*4 - 32

        scale = (header[14] & 0xc0) >> 6
        field = 2 if header[10] & 0x80 > 0 else 0
        channel_id = (header[10] & 0x78) >> 3

        tmp_bit = header[4] & 0x7f
        year = tmp_bit + 2000 if tmp_bit < 128 else 0
        month = (header[5] & 0xf0) >> 4 if tmp_bit != 0xff else 0
        date = (header[6] & 0xf8) >> 3

        tmp_word = int.from_bytes(header[6:8], 'big')
        hour = (tmp_word & 0x07c0) >> 6
        minute = (header[8] & 0xfc) >> 2

        tmp_word = int.from_bytes(header[8:10], 'big')
        sec = (tmp_word & 0x03f0) >> 4

        t = time.mktime((year, month, date, hour, minute, sec, 0, 0, 0))
        
        return {
            'jpeg_size': jpeg_size,
            'scale': scale,
            'field': field,
            'channel_id': channel_id,
            'time': t,
            'bytes': np.frombuffer(header, dtype=np.uint8)
        }

    def read_frame(buf, header=None):
        if not header: header = read_header(buf)
        if not header: return

        if header['scale'] == 0:
            x_scale = 720
            offset = Rect(10, 10-header['field'], 10, 10+header['field'])
        elif header['scale'] == 1:
            x_scale = 640
            offset = Rect(10, 10-header['field'], 10, 10+header['field'])
        elif header['scale'] == 2:
            x_scale = 368
            offset = Rect(5, 6-header['field'], 5, 6+header['field'])
        elif header['scale'] == 3:
            x_scale = 320
            offset = Rect(5, 6-header['field'], 5, 6+header['field'])
        else:
            return
        
        def swap_bytes(basestring):
            a2 = np.frombuffer(basestring, dtype=np.int32)
            a2 = a2.byteswap()
            a2.dtype = np.int16
            a2 = a2.byteswap()
            a2[-2] = 0
            a2.dtype = np.uint8
            return a2
        
        height = 240 - offset.bottom - offset.top
        width = x_scale - offset.left - offset.right

        im_buf = np.empty(width*height*100, dtype=ctypes.c_ubyte)
        data = np.concatenate((header['bytes'], swap_bytes(buf.read(header["jpeg_size"]))), dtype=ctypes.c_ubyte)
        
        error = nvp.NVP3000_MJPEG(data, im_buf, 0, 0, 0, 128, 0, 0, 128)
        
        if error: 
            raise Exception('Failed to create JPEG')
        
        img = np.ctypeslib.as_array(im_buf[(offset.bottom*x_scale)*3:(height+offset.bottom)*x_scale*3]).reshape(height, -1, 3)
        
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (21, 7), 0)
        
        c_id = header['channel_id']
        if c_id not in cams:
            cams[c_id] = {}
            cams[c_id]['ref_frame'] = gray
            cams[c_id]['diff'] = []
            cams[c_id]['time'] = []
        else:
            frameDelta = cv.absdiff(cams[c_id]['ref_frame'], gray)
            thresh = cv.threshold(frameDelta, 20, 255, cv.THRESH_BINARY)[1]
            cams[c_id]['diff'].append(np.sum(thresh)/width/height)
            cams[c_id]['time'].append(header['time'])
            cams[c_id]['ref_frame'] = gray
        
    cams = {}

    with open(file, 'rb') as f:
        print(f'Processing {file}...')

        while True:
            bit = f.read(1)
            if bit == b'': break
            if bit == b'\xff':
                f.seek(15, 1)
                header = read_header(f)
                if (header):
                    read_frame(f, header)

    dfs = []
    for i in range(4):
        dfi = pd.DataFrame({'time': cams[i]['time'], f'Cam. {i+1}': cams[i]['diff']})
        dfi.set_index('time', inplace=True)
        dfi.set_index(pd.to_datetime(dfi.index, unit='s') + pd.to_timedelta(dfi.groupby(level=0).cumcount(), unit='ms'), 
                    inplace=True)
        dfi = dfi.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
        dfs.append(dfi)

    df = pd.concat(dfs, axis=1)
    df[df.isna()] = 0
    df.index = df.index.tz_localize('utc').tz_convert('Europe/Minsk').strftime('%H:%M:%S')

    out = Path('out/')
    out.mkdir(exist_ok=True)
    
    df.loc[(df > 0.1).any(axis=1)].to_excel(f'{out/path.stem}.xlsx')

if __name__ == '__main__':
    start_time = time.time()
    try:
        with Executor(max_workers=args.nproc) as executor:
            result = list(executor.map(proc_file, args.files))
    except Exception as ex:
        print('An error occurred')
        raise ex

    print(f'Done! Elapsed time: {round(time.time()-start_time, 2)}s')