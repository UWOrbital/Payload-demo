import numpy as np
from PIL import Image

from decoder import decode

args = decode.decoder_arguments()
with open(args.codes_filename, 'rb') as codes_file:
    decode.load_codes(codes_file)
with open(args.filename, 'rb') as data_file:
    data_stream = decode.Bitstream(data_file)
    img_arr = decode.decode(data_stream, args.mcu_count)
img_arr = decode.trim_domain(img_arr)
img = Image.fromarray(img_arr.astype(np.uint8))
img.save(f'{args.filename}_decoded.jpg')
img.show()
